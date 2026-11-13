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
  * every ratio the text quotes has a prose twin <name>Pr (two significant figures, the direction
    in words), and a ratio involving a two-run configuration also <name>PrMax (the value farther
    from 1; <name>Pr is the one nearer, the value less favourable to a claimed difference).

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

import collections
import csv
import importlib.util
import json
import math
import pathlib
import re
import statistics

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
         "KIN": r"\Kfull{} classes, kinematic inputs", RAND_POOLED: "random partitions",
         SOPHON: "released Sophon checkpoint",
         **{f"RAND2_p{k}": f"random partition {k}" for k in range(1, 6)}}
SCRATCH_LABEL = V4.SCRATCH_LABEL      # "from scratch": a fine-tuning table's reference, never mode-tagged
HEADING = {"ft": "fine-tuned", "lp": "linear probe on the frozen features",
           "mlp": "multilayer-perceptron probe on the frozen features", "head": "mass output of the pretraining head"}
PARTITIONS = "17-class partitions"      # the subheading over the indented partition rows


def _full_row(ncol: int, text: str, italic: bool = False) -> str:
    """A heading across a tabular of ncol columns, flush with the row labels."""
    return rf"\multicolumn{{{ncol}}}{{@{{}}l}}{{" + (rf"\textit{{{text}}}" if italic else text) + r"} \\"


FITS, MLP_FITS = "linprobe/fits_v3", "linprobe/fits_v3_mlp"
MLP_FITS_V4, FITS_INITBN, MLP_FITS_INITBN = ("linprobe/fits_v3_mlp2", "linprobe/fits_v3_initbn",
                                             "linprobe/fits_v3_mlp2_initbn")
FITS_FILL, MLP_BAL = "linprobe/fits_v4_fill", "linprobe/fits_v4_mlp"
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
            ("jc2_qcd", "leg188_leg1_metrics", None, "macro_auc_qcd", "JetClass-II, QCD classes"),
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


class Bound(float):
    """A rejection at the sample's bound (no background jet passed): a lower limit."""


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
    return head + r"\%" + (sp + tail if sp else "")


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
    return Bound(rec[k]) if rec.get(f"{k}_is_bound") else rec[k]


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


# ------------------------------------------------------------------ ratios and their prose twins
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


def _within(dev: float) -> str:
    """'equal within X%': the deviation in per cent rounded up to one significant figure, at least 1."""
    x = max(100 * dev, 1.0)
    e = math.floor(math.log10(x))
    m = math.ceil(x / 10 ** e - 1e-9)
    return f"equal within {m * 10 ** e:g}\\%"


def _bounded(r: float, noun: str, nearer: bool) -> str:
    """One of two runs in words: the value nearer 1 as the weakest effect ('at least ...', or for a
    reduction written as a factor 'at most ...'), the farther one as the strongest ('up to ...')."""
    ph = _phrase(r, noun)
    if r < 1 and " times the " in ph:
        return ("at most " if nearer else "down to ") + ph
    return ("at least " if nearer else "up to ") + ph


def prose(vals: list, noun: str, farther: bool = False, ref_sd: float | None = None) -> str:
    """The prose twin of a ratio: three or more runs, the mean in words unless it lies within twice
    the SD of 1 ('equal within X%'); two runs, the value nearer 1 (farther=True: the other), unless
    they straddle 1 or, given ref_sd (the SD of the same ratio between levels of the class hierarchy,
    _level_sd), either lies within twice ref_sd of 1 on the scale _level_sd uses, the logarithm (the
    Methods rule for two runs); one run, its value."""
    v = [float(x) for x in vals if x is not None]
    if not v:
        return r"\pending{no run yet}"
    if len(v) >= 3:
        m, sd = float(np.mean(v)), float(np.std(v, ddof=1))
        return _within(max(abs(m - 1), sd)) if abs(m - 1) <= 2 * sd else _phrase(m, noun)
    if len(v) == 2:
        a, b = sorted(v, key=lambda x: abs(math.log(x)) if x > 0 else math.inf)
        if (a - 1) * (b - 1) < 0 or (ref_sd is not None and a > 0 and abs(math.log(a)) <= 2 * ref_sd):
            return _within(max(abs(a - 1), abs(b - 1)))
        return _bounded(b if farther else a, noun, not farther)
    return _phrase(v[0], noun)


def _pct_prose(vals: list) -> str:
    """A per-cent value in prose: two significant figures of the mean; two runs, both."""
    v = [float(x) for x in vals if x is not None]
    if not v:
        return r"\pending{no run yet}"
    if len(v) == 2:
        return f"{_sig2(min(v))}--{_sig2(max(v))}\\%"
    return f"{_sig2(float(np.mean(v)))}\\%"


def _emit_ratio(em, name: str, vals: list, n_expected: int, fmt_pm, src, path: str, noun: str,
                two_run: bool = False, body: str | None = None, ref_sd: float | None = None) -> None:
    """A ratio macro, its prose twin and, for a two-run configuration, the twin of the other run;
    ref_sd, when given, is the hierarchy-level SD the prose twins read two runs against (prose)."""
    em.macro(name, body if body is not None else _cell(vals, n_expected, fmt_pm, sep=TEXT_SEP), src, path)
    em.macro(name + "Pr", prose(vals, noun, ref_sd=ref_sd), src, path, "prose twin: two significant figures, direction in words")
    if two_run:
        em.macro(name + "PrMax", prose(vals, noun, farther=True, ref_sd=ref_sd), src, path,
                 "prose twin of the run farther from 1")


def _undefined(em, name: str, src, path: str, two_run: bool = False) -> None:
    """A ratio of 1 - AUC whose reference separates the test jets completely (1 - AUC = 0): a dash,
    and prose twins that say so."""
    em.macro(name, "--", src, path, "the reference's 1 - AUC is 0")
    for t in ("Pr", "PrMax") if two_run else ("Pr",):
        em.macro(name + t, "undefined, since the reference separates the test jets completely", src, path,
                 "the reference's 1 - AUC is 0")


def _emit_pct(em, name: str, vals: list, n_expected: int, fmt_pm, src, path: str) -> None:
    em.macro(name, _pct_cell(vals, n_expected, fmt_pm), src, path)
    em.macro(name + "Pr", _pct_prose(vals), src, path, "prose twin, two significant figures")


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


# the ratios between levels of the class hierarchy whose SD a two-run configuration is read against
LEVEL_RATIOS = (("L162", "L188"), ("R42_Q1", "L188"), ("R16_Q1", "L188"))


def _level_sd(runs, value) -> float | None:
    """The SD of a ratio between levels of the class hierarchy (Methods, the rule for two runs): of
    value, each coarser level over the 188-class run of the same index, the SD over the runs of the
    ratio's logarithm (its relative SD, so that a level whose ratio is far from 1 does not widen the
    threshold of one near 1 by its scale alone), pooled over the three levels (root mean square);
    None while no level has two runs."""
    var = []
    for arm, ref in LEVEL_RATIOS:
        r = [math.log(x) for x in _ratio_vals(runs, arm, ref, value)[0] if x is not None and x > 0]
        if len(r) >= 2:
            var.append(float(np.var(r, ddof=1)))
    return math.sqrt(float(np.mean(var))) if var else None


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
    reference's spread, and the prose twins from the per-run ratios."""
    if not ref:
        em.macro(name, r"\pending{reference}", src, path)
        em.macro(name + "Pr", r"\pending{reference}", src, path)
        if two_run:
            em.macro(name + "PrMax", r"\pending{reference}", src, path)
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
    rec = _probe_rec(fits, KIN_MODEL, "top") if kin else None
    if rec is not None:         # after the pretrained rows, before the random network
        out.append(LABEL["KIN"] + " & " + " & ".join([_cell([_rej(rec, k) if k.startswith("r") else rec[k]], 1, fmt_pm, frac=not k.startswith("r"))
                                                  for k, _ in BENCH4] + ["--"] * 2 * len(BENCH4)) + r" \\")
    out += [lower(arm) for arm in rows if arm in (INIT, INIT_BN)]
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




BLOCK_WHOLE = 24    # rows of a task block kept on one turned page (it holds about 30 lines at \\small)


def _task_longtable(blocks: list, label: str) -> str:
    """One longtable of task blocks [(title, [heading cells, row cells, ...])]: each block opens with
    its title across the table and its own heading row; a row may have fewer cells than the widest
    block (the rest stay empty). Set in landscape across the whole line at the tables' \\small, the row labels on
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
    return "\n".join([r"\begin{landscape}", "{\\small", LT_SETUP,
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
# per model (scratch-s<k> from scratch) with "curves": {pair: {auc, ...}}; the JetClass-II fine-tuning at 1e6 scored
# with g_a / (g_a + g_b) on the pair's jets among the first 2,000,000 TEST2M jets, natural composition (the jets of the
# head sculpting, scripts/build_head_jobs.py --ft-head)
PAIRS_FT = "curves_v2/pairs_ft"
PAIRS_LABEL_CM = 4.6        # the pair table's row labels wrap at this width and its blocks may be shrunk by
                            # SHRINK_OK, so that two pairs fit a block
FT_SCRATCH_MODELS = ("scratch-s1", "scratch-s2", "scratch-s3")


def load_pairs_ft(data: pathlib.Path) -> dict:
    """{model: {pair: record}} of the fine-tuned pair read-outs, {} until pulled."""
    return {m: d.get("curves", d) for m, d in _load(data / PAIRS_FT).items()}


def table_pairs(fits, runs, fmt_pm, status, rows, ft=None, rho=True) -> str:
    """A11, binary probes between JetClass-II classes: the AUC, and the ratio of 1 - AUC to the
    188-class run as a labelled column ('--' where that run separates the test jets completely);
    a dagger marks a pair the label set merges. ft (load_pairs_ft; {} prints the group pending) adds the fine-tuned
    networks as a second row group: the mode in each group's heading and not in the row labels, a row fine-tuned
    from scratch (no probe) and none for the random networks (no fine-tuning). One tabular, the pairs as column
    groups (split_wide deals them into blocks). rho=False (the paper, PI 2026-11-12): the AUC only, named once in
    the heading over the pairs. The pooled random partitions (RAND_POOLED) are marked where some but not all of the
    five merge the pair (POOL_MARK). A mean AUC that rounds to 1.0000 prints as NEAR_ONE and takes no part in its
    column's precision."""
    mark = {"merged": r"$^{\dagger}$", "partly": r"$^{\ddagger}$", "apart": "", "some": POOL_MARK}
    if RAND_POOLED in rows:
        def pooled(key):
            st = [status.get(a, {}).get(key, "apart") for a in RAND]
            return "merged" if all(x == "merged" for x in st) else "apart" if all(x == "apart" for x in st) else "some"
        status = {**status, RAND_POOLED: {key: pooled(key) for key, *_ in PAIRS}}
    modes = [("lp", tuple(rows), lambda m, key: _pair(fits, m, key))]
    if ft is not None:          # until pulled ({}), one pending row in place of the group's rows
        modes.append(("ft", tuple(r for r in rows if r not in (INIT, INIT_BN)) + (SCRATCH,) if ft else (),
                      lambda m, key: (ft.get(m, {}).get(key) or {}).get("auc")))
        runs = {**runs, SCRATCH: list(FT_SCRATCH_MODELS)}
    body = {}                                       # (mode, arm) -> [(AUC cell, rho cell)] per pair
    for mode, arms, get in modes:
        for arm in arms:
            body[mode, arm] = []
            for key, *_ in PAIRS:
                v = [get(m, key) for m in runs[arm]]
                near = None not in v and round(float(np.mean(v)), 4) >= 1
                auc = ((NEAR_ONE + SINGLE * (len(v) == 1) if near else _cell(v, len(runs[arm]), fmt_pm, frac=True))
                       + (mark[status.get(arm, {}).get(key, "apart")] if mode == "lp" else ""))
                r, n = _ratio_to_full(fits, runs, arm, lambda m: _oma(get(m, key)))
                sep = separated(fits, runs, key, get) & {_idx(m) for m in runs[arm]}
                body[mode, arm].append((auc, _cell(r, n, fmt_pm) if n and not sep else "--"))
    # a pair whose rho is a dash in every row (the 188-class runs separate it) shows its AUC only
    has_rho = [rho and any(c[i][1] != "--" for c in body.values()) for i in range(len(PAIRS))]
    w = [2 if h else 1 for h in has_rho]
    out = [r"\begin{tabular}{l" + "c" * sum(w) + "}", r"\toprule"]
    if rho:
        out += [" & " + " & ".join(rf"\multicolumn{{{k}}}{{c}}{{{h}}}" for k, (_, h, *_) in zip(w, PAIRS)) + r" \\",
                "".join(rf"\cmidrule(lr){{{2 + sum(w[:i])}-{1 + sum(w[:i + 1])}}}" for i in range(len(w))),
                " & " + " & ".join(r"AUC & $\rho$" if h else "AUC" for h in has_rho) + r" \\", r"\midrule"]
    else:
        out += [rf" & \multicolumn{{{len(PAIRS)}}}{{c}}{{AUC}} \\", rf"\cmidrule(lr){{2-{1 + len(PAIRS)}}}",
                " & " + " & ".join(h for _, h, *_ in PAIRS) + r" \\", r"\midrule"]
    for k, (mode, arms, _) in enumerate(modes):
        if len(modes) > 1:
            out += ([r"\midrule"] if k else []) + [_full_row(1 + sum(w), HEADING[mode], italic=True)]
        if not arms:
            out.append(_full_row(1 + sum(w), r"\pending{no run yet}"))
        for arm in arms:
            out.append((SCRATCH_LABEL if arm == SCRATCH else _row(arm, runs)) + " & " + " & ".join(
                a + (" & " + r if h else "") for (a, r), h in zip(body[mode, arm], has_rho)) + r" \\")
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
    partitions that merge the pair over that of those that keep it apart; each with its prose twin."""
    for ds, met, _ in JC2_TASKS:
        n = _largest(fits, ds)
        w = TASK_WORD[(ds, met)]
        f = lambda m, ds=ds, n=n, met=met: _oma(_probe(fits, m, ds, n, met))
        sd = _level_sd(runs, f)
        for arm, ref in CMP_PROBES:
            r, k = _ratio_vals(runs, arm, ref, f)
            _emit_ratio(em, f"MlstCmp{w}{WORD[arm]}Over{WORD[ref]}", r, k, fmt_pm, index,
                        f"linear probe {ds} N={n}: (1 - {met}) of {arm} over {ref}, run by run", "errors",
                        _two_run(runs, arm, ref), ref_sd=sd)
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
        sd = _level_sd(runs, f)
        for arm, ref in CMP_PAIRS:
            name, path = (f"MlstCmp{PAIR_WORD[key]}{WORD[arm]}Over{WORD[ref]}",
                          f"binary probe {key}: (1 - AUC) of {arm} over {ref}, run by run")
            if zero(ref, (arm,)):
                _undefined(em, name, index, path, _two_run(runs, arm, ref))
                continue
            r, k = _ratio_vals(runs, arm, ref, f)
            _emit_ratio(em, name, r, k, fmt_pm, index, path, "errors", _two_run(runs, arm, ref), ref_sd=sd)
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
           r" & \multicolumn{2}{c}{natural composition} & \multicolumn{2}{c}{class-balanced} \\",
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
    bal = _load(data / MLP_BAL)
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


def table_mlp(mlp, runs, fmt_pm, rows, raw_v4=None, nat=None, kin=False) -> str:
    """A5: the nonlinear probe (one hidden layer, three seeds averaged) at the largest training set:
    JetClass-II AUC (all, resonant, QCD) with the ratio of 1 - AUC to the 188-class run; JetClass AUC;
    top, quark/gluon and Herwig AUC and R30; the class pairs' AUC and ratio; for the configurations with
    no class-balanced fit (their cells above a dash), JetClass-II and the pairs' ratios on natural
    composition (nat, load_mlp_natural) beside the levels of the hierarchy on the same test set, so that no
    ratio crosses the two compositions; and the linear probe on the head's 512-unit hidden layer beside
    the one on the class token. kin=True adds the kinematic-only model to the first block (top only)."""
    nat_only = [arm for arm in rows if nat and all(_mlp_val(mlp, m, "jc2_188") is None for m in runs[arm])
                and any(_mlp_val(nat, m, "jc2_188") is not None for m in runs[arm])]
    out = [r"\begin{tabular}{l" + "c" * 11 + "}", r"\toprule",
           r" & \multicolumn{4}{c}{JetClass-II} & JetClass & \multicolumn{2}{c}{top} & \multicolumn{2}{c}{quark/gluon}"
           r" & \multicolumn{2}{c}{Herwig} \\",
           r"\cmidrule(lr){2-5}\cmidrule(lr){7-8}\cmidrule(lr){9-10}\cmidrule(lr){11-12}",
           r" & AUC & $\rho$ & AUC, resonant & AUC, QCD & AUC & AUC & "
           r"$R_{30}$ & AUC & $R_{30}$ & AUC & $R_{30}$ \\", r"\midrule"]
    for arm in rows:
        ms, n = runs[arm], len(runs[arm])
        if arm in nat_only:
            cells = ["--"] * 4
        else:
            r, k = _ratio_to_full(None, runs, arm, lambda m: _mlp(mlp, m, "jc2_188"))
            cells = [_cell([_mlp_val(mlp, m, t) for m in ms], n, fmt_pm, frac=True) for t in ("jc2_188", "jc2_188_res", "jc2_188_qcd")]
            cells.insert(1, _cell(r, k, fmt_pm) if k else "--")       # beside the AUC it is the ratio of
        cells.append(_cell([_mlp_val(mlp, m, "jc1") for m in ms], n, fmt_pm, frac=True))
        for t in ("top", "qg", "qg_herwig"):
            cells += [_cell([_mlp_val(mlp, m, t) for m in ms], n, fmt_pm, frac=True),
                      _cell([_mlp_val(mlp, m, t, "r30") for m in ms], n, fmt_pm)]
        out.append(_row(arm, runs) + " & " + " & ".join(cells) + r" \\")
    if kin and _mlp_val(mlp, KIN_MODEL, "top") is not None:     # top only, nothing missing, one run
        out.append(LABEL["KIN"] + " & " + " & ".join(["--"] * 5 + [_cell([_mlp_val(mlp, KIN_MODEL, "top")], 1, fmt_pm, frac=True),
                                                            _cell([_mlp_val(mlp, KIN_MODEL, "top", "r30")], 1, fmt_pm)]
                                                + ["--"] * 4) + r" \\")
    out += _hidden_rows(raw_v4, runs, fmt_pm) + [r"\bottomrule", r"\end{tabular}"]
    for part in (MLP_PAIRS[:3], MLP_PAIRS[3:]):
        out += ["%SPLIT%", r"\begin{tabular}{l" + "cc" * len(part) + "}", r"\toprule",
                " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{h}}}" for _, h, _, _ in part) + r" \\",
                "".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(part))),
                " & " + " & ".join([r"AUC & $\rho$"] * len(part)) + r" \\",
                r"\midrule"]
        for arm in (a for a in rows if a not in nat_only):
            cells = []
            for key, _, _, _ in part:
                cells.append(_cell([_mlp_val(mlp, m, key) for m in runs[arm]], len(runs[arm]), fmt_pm, frac=True))
                r, k = _ratio_to_full(None, runs, arm, lambda m, key=key: _mlp(mlp, m, key))
                cells.append(_cell(r, k, fmt_pm) if k else "--")
            out.append(_row(arm, runs) + " & " + " & ".join(cells) + r" \\")
        out += [r"\bottomrule", r"\end{tabular}"]
    if nat_only:
        out += ["%SPLIT%", r"\begin{tabular}{l" + "c" * (2 + len(MLP_PAIRS)) + "}", r"\toprule",
                r" & \multicolumn{2}{c}{JetClass-II} & \multicolumn{" + str(len(MLP_PAIRS))
                + r"}{c}{$\rho$ on the class pairs} \\",
                r"\cmidrule(lr){2-3}\cmidrule(lr){4-" + str(3 + len(MLP_PAIRS)) + "}",
                r"natural-composition test set & AUC & $\rho$ & " + " & ".join(h for _, h, _, _ in MLP_PAIRS) + r" \\", r"\midrule"]
        for arm in MAIN + tuple(nat_only):
            cells = [_cell([_mlp_val(nat, m, "jc2_188") for m in runs[arm]], len(runs[arm]), fmt_pm, frac=True)]
            for t in ("jc2_188", *(k for k, *_ in MLP_PAIRS)):
                r, k = _ratio_to_full(None, runs, arm, lambda m, t=t: _mlp(nat, m, t))
                cells.append(_cell(r, k, fmt_pm) if k else "--")
            out.append(_row(arm, runs) + " & " + " & ".join(cells) + r" \\")
        out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out + [""])


def _hidden_rows(raw_v4, runs, fmt_pm) -> list:
    """The linear probe on the pretraining head's 512-unit hidden layer beside the one on the class token (188 classes),
    as rows under a rule at the foot of the nonlinear-probe table's first block, in its JetClass-II AUC and quark/gluon
    columns (a dash elsewhere): a block of its own was two rows alone on a page (PI 2026-11-11)."""
    hid = [m for m in runs["L188"] if "jc2bal.hidden" in (raw_v4 or {}).get(m, {}).get("cells", {}).get(PRIMARY, {})]
    if not hid:
        return []

    def cell(m, key, met):
        c = raw_v4[m]["cells"][PRIMARY].get(key, {})
        c = c[max(c, key=int)] if c else None
        return None if c is None else (_rej(c, met) if met.startswith("r") else c[met])
    out = [r"\midrule"]
    for name, sfx in ((LABEL["L188"] + ", linear probe, class token", ""), (LABEL["L188"] + ", linear probe, 512-unit hidden layer", ".hidden")):
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
        sd = _level_sd(runs, f)
        for arm, ref in CMP_MLP:
            r, k = _ratio_vals(runs, arm, ref, f)
            _emit_ratio(em, f"MlstMlpCmp{PAIR_WORD[t]}{WORD[arm]}Over{WORD[ref]}", r, k, fmt_pm, index,
                        f"nonlinear probe {t}: (1 - AUC) of {arm} over {ref}, run by run", "errors", _two_run(runs, arm, ref),
                        ref_sd=sd)
        r = _rand_over(runs, "L188", f)
        _emit_ratio(em, f"MlstMlpCmp{PAIR_WORD[t]}RandOverFull", r, len(r), fmt_pm, index,
                    f"nonlinear probe {t}: (1 - AUC) of each random partition over the 188-class run of its seed", "errors")
    for t in ("top", "qg", "qg_herwig"):
        f = lambda m, t=t: _mlp_val(mlp, m, t, "r30")
        sd = _level_sd(runs, f)
        for arm in MAIN:
            r, k = _ratio_vals(runs, arm, INIT_BN, f)
            _emit_ratio(em, f"MlstMlp{PAIR_WORD[t]}Rthirty{WORD[arm]}OverInitBn", r, k, fmt_pm, index,
                        f"nonlinear probe {t}, missing inputs zero: R30 of {arm} over the BatchNorm-recomputed random "
                        "network of the same index", "rejection", ref_sd=sd)


def mlp_natural_macros(em, nat, runs, index, fmt_pm) -> None:
    """MlstMlpNatJcTwo<mass>Over<same labels>: the nonlinear probe's 1 - AUC on natural-composition
    JetClass-II with over without the auxiliary mass regression, run by run, read against the
    hierarchy-level SD on the same test set; these models have no class-balanced nonlinear probe."""
    f = lambda m: _mlp(nat, m, "jc2_188")
    sd = _level_sd(runs, f)
    for arm, ref in (("L162_MASS", "L162"), ("R16_Q1_MASS", "R16_Q1")):
        r, k = _ratio_vals(runs, arm, ref, f)
        _emit_ratio(em, f"MlstMlpNatJcTwo{WORD[arm]}Over{WORD[ref]}", r, k, fmt_pm, index,
                    f"nonlinear probe, natural-composition JetClass-II (fits_v3_mlp2): (1 - AUC) of {arm} over {ref}, run by run",
                    "errors", _two_run(runs, arm, ref), ref_sd=sd)


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


def _bench_rec(fits: dict, fill: dict, chosen: dict, m: str, ds: str, n: str | None = None):
    """A top or quark/gluon probe cell at size n (default the largest) under the substitution chosen
    on validation (fits_v4_fill), or with the missing inputs zero (fits) while the choice is pending.
    The kinematic-only model has nothing missing and is read from fits; the released Sophon checkpoint has no
    substitution probes, so its missing inputs stay zero (the captions say so)."""
    s = chosen.get(ds)
    if s is None or m in (KIN_MODEL, SOPHON_MODEL):
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
    return f"missing inputs: {SUBST_WORD[s]}" if s else r"missing inputs: zero \pending{choice on validation}"


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
    ({task: substitution}, the paper): that substitution's column is left out, and the kinematic-only row with it:
    tab:probes prints both, and a number has one table."""
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
            rf" & \multicolumn{{{k1}}}{{c}}{{top, $R_{{30}}$}} & \multicolumn{{{k2}}}{{c}}{{quark/gluon, $R_{{30}}$}} \\",
            rf"\cmidrule(lr){{2-{1 + k1}}}\cmidrule(lr){{{2 + k1}-{1 + k1 + k2}}}",
            " & " + " & ".join(subs["top"] + subs["qg"]) + r" \\", r"\midrule"]
    kin = _rej(_probe_rec(fits, KIN_MODEL, "top"), "r30")
    kin_row = None if chosen else (
        LABEL["KIN"] + r" & \multicolumn{3}{c}{" + (_lone(kin) + SINGLE if kin is not None else r"\pending{run}")
        + r" (nothing missing)} & -- & -- & -- \\")
    for arm in rows:
        ms = runs[arm]
        cells = []
        for t in ("top", "qg"):
            cells += [_cell([_fill_r30(fill, m, t, s) for m in ms], len(ms), fmt_pm) for s in subs[t]]
        if arm in (INIT, INIT_BN) and kin_row:      # after the pretrained rows, before the random network
            out.append(kin_row)
            kin_row = None
        out.append(_row(arm, runs) + " & " + " & ".join(cells) + r" \\")
    if kin_row:
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
        sds = {sfx: _level_sd(runs, f) for f, sfx in ((sel, ""), (zero, "Zero"))}
        for arm in MAIN + MASS:
            for f, sfx in ((sel, ""), (zero, "Zero")):
                r, k = _ratio_vals(runs, arm, INIT_BN, f)
                _emit_ratio(em, f"MlstLp{tw}Rthirty{WORD[arm]}OverInitBn{sfx}", r, k, fmt_pm, index,
                            f"linear probe {task}, largest set, {'chosen substitution' if not sfx else 'zero'}: R30 of {arm} "
                            "over the BatchNorm-recomputed random network of the same index", "rejection",
                            _two_run(runs, arm), ref_sd=sds[sfx])
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
    out = [r"\begin{tabular}{lcc}", r"\toprule", r" & resolution & median response \\", r"\midrule"]
    out.append("soft-drop mass & " + " & ".join(
        _cell([v], 1, fmt_pm, run=False) if (v := _groomed(atlas, g, k)) is not None else "--" for k in ("resolution", "median")) + r" \\")
    for est, arms in (("regressed", MASS), ("mlp", MASS_ROWS)):
        out += [r"\midrule", _full_row(3, HEADING["head" if est == "regressed" else "mlp"], italic=True)]
        for arm in arms:
            ms = runs[arm]
            out.append(LABEL[arm] + " & " + " & ".join(_cell([_ares(atlas, m, est, g, k) for m in ms], len(ms), fmt_pm)
                                                       for k in ("resolution", "median")) + r" \\")
    return "\n".join(out + [r"\bottomrule", r"\end{tabular}", ""])


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


def table_mass_detail(atlas, runs, fmt_pm) -> str:
    """tab:massdetail (supplement): the values of the binned mass figures and the inclusive values Table 5 (tab:mass)
    does not hold. Upper part: over resonant and QCD jets, the soft-drop mass, the mass output of the pretraining head
    and the multilayer-perceptron probe; the ridge regression over all three jet groups. Lower parts: resolution and
    median response in the bins of jet pT and generator mass that the figures draw (atlas: atlas_res_fine, ten pT
    bins), per MASS_DETAIL_BINNED block, the soft-drop mass once over each jet group."""
    sec = lambda t, k: r"\multicolumn{" + str(k) + r"}{@{}l}{\textit{" + t + r"}} \\"   # noqa: E731
    group = {g: h for g, _, h in MASS_GROUPS}
    out = []
    for groups, blocks in ((("resonant", "qcd"), (("regressed", HEADING["head"], MASS), ("mlp", HEADING["mlp"], MASS_ROWS))),
                           (("all", "resonant", "qcd"), (("ridge", RIDGE, MASS_ROWS),))):
        ng = len(groups)
        out += [r"\begin{tabular}{l" + "c" * 2 * ng + "}", r"\toprule",
                rf" & \multicolumn{{{ng}}}{{c}}{{resolution}} & \multicolumn{{{ng}}}{{c}}{{median response}} \\",
                rf"\cmidrule(lr){{2-{1 + ng}}}\cmidrule(lr){{{2 + ng}-{1 + 2 * ng}}}",
                "jets & " + " & ".join(group[g] for _ in range(2) for g in groups) + r" \\", r"\midrule"]
        if "all" not in groups:
            out.append("soft-drop mass & " + " & ".join(_cell([v], 1, fmt_pm, run=False) if (v := _groomed(atlas, g, k)) is not None
                                                        else "--" for k in ("resolution", "median") for g in groups) + r" \\")
        for est, name, arms in blocks:
            out.append(sec(name, 1 + 2 * ng))
            for arm in arms:
                ms = runs[arm]
                out.append(f"{LABEL[arm]} & " + " & ".join(
                    _cell([_ares(atlas, m, est, g, k) for m in ms], len(ms), fmt_pm)
                    for k in ("resolution", "median") for g in groups) + r" \\")
        out += [r"\bottomrule", r"\end{tabular}", "%SPLIT%"]
    out.pop()
    for axis, title in (("vs_pt", r"jet $p_{\mathrm{T}}$ [GeV]"), ("vs_mgen", r"generator-level soft-drop mass [GeV]")):
        edges = _binned_one(atlas, axis)
        if not edges:
            continue
        bins = [f"{a:g}--{b:g}" for a, b in zip(edges, edges[1:])]
        for key, kname in (("resolution", "resolution"), ("median", "median response")):
            out += ["%SPLIT%", r"\begin{tabular}{l" + "c" * len(bins) + "}", r"\toprule",
                    rf"\textit{{{kname}}} & \multicolumn{{{len(bins)}}}{{c}}{{{title}}} \\",
                    rf"\cmidrule(lr){{2-{1 + len(bins)}}}",
                    " & " + " & ".join(bins) + r" \\", r"\midrule"]
            for est, g, arms in MASS_DETAIL_BINNED:
                name = {"regressed": HEADING["head"], "mlp": HEADING["mlp"], "ridge": RIDGE}[est]
                # upright, so that the turned longtable runs on through its sections rather than starting a page at each
                # (_turned_long starts one at every italic heading after the first)
                out.append(_full_row(1 + len(bins), f"{name}, {group[g]} jets"))
                if est == "regressed":          # the soft-drop mass once a jet group, beside the mass outputs
                    gv = _groomed(atlas, g, key, axis) or []
                    out.append("soft-drop mass & " + " & ".join("--" if x is None else _cell([x], 1, fmt_pm, run=False) for x in gv) + r" \\")
                for arm in arms:
                    ms = runs[arm]
                    per = [_ares(atlas, m, est, g, key, axis) for m in ms]
                    out.append(f"{LABEL[arm]} & " + " & ".join(
                        _cell([None if p is None else p[i] for p in per], len(ms), fmt_pm) for i in range(len(bins))) + r" \\")
            out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out + [""])


CMP_MASS = (("L162", "L188"), ("R42_Q1", "L188"), ("R16_Q1", "L188"), ("L162_MASS", "L162"),
            ("R16_Q1_MASS", "R16_Q1"))


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
        sd = _level_sd(runs, f)
        for arm, ref in CMP_MASS:
            r, k = _ratio_vals(runs, arm, ref, f)
            _emit_ratio(em, f"MlstCmpMlp{WORD[arm]}Over{WORD[ref]}{gs}", r, k, fmt_pm, index,
                        f"frozen-feature MLP resolution of {arm} over {ref}, run by run, {g} jets", "resolution",
                        _two_run(runs, arm, ref), ref_sd=sd)
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


def table_sculpt(sculpt, runs, fmt_pm, rows) -> str:
    """A14 with M2 (PI 2026-11-07): at each QCD efficiency, the Jensen-Shannon distance (x 10^2) between the
    soft-drop mass distributions of all QCD test jets and of those passing the cut on 1 - p_QCD, and the share of
    resonant test jets passing the same cut; at the cut keeping half of the resonant jets (DisCo's working point),
    the QCD rejection R50 and the distance there. Below the runs the distance a random selection of as many QCD
    jets shows. The 50% columns need the files of mass_sculpting.py with sig_eff_50."""
    v3 = all("sig_eff_50" in d for d in sculpt.values())
    lines = [r"\begin{tabular}{l" + "cc" * len(SCULPT_EFFS) + ("cc" if v3 else "") + "}", r"\toprule",
             " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{$\epsB = {100 * float(e):g}\%$}}" for e in SCULPT_EFFS)
             + (r" & \multicolumn{2}{c}{$\varepsilon_S = 50\%$}" if v3 else "") + r" \\",
             "".join(r"\cmidrule(lr){%d-%d}" % (2 + 2 * q, 3 + 2 * q) for q in range(len(SCULPT_EFFS) + v3)),
             " & " + " & ".join([r"distance & $\varepsilon_S$"] * len(SCULPT_EFFS))
             + (r" & $R_{50}$ & distance" if v3 else "") + r" \\", r"\midrule"]
    eff_s = lambda m, e: None if (d := sculpt.get(m)) is None else d["working_points"][e]["eff_sig"]
    at50 = lambda m, k: None if (d := sculpt.get(m)) is None else d["sig_eff_50"][k]
    for arm in rows:
        ms, n = runs[arm], len(runs[arm])
        cells = []
        for e in SCULPT_EFFS:
            cells += [_cell([_sculpt(sculpt, m, e) for m in ms], n, fmt_pm, 1e2),
                      _cell([eff_s(m, e) if v3 else None for m in ms], n, fmt_pm, frac=True) if v3 else "--"]
        if v3:
            cells += [_cell([at50(m, "rejection") for m in ms], n, fmt_pm),
                      _cell([None if (x := at50(m, "jsd")) is None else math.sqrt(x) for m in ms], n, fmt_pm, 1e2)]
        lines.append(LABEL[arm] + " & " + " & ".join(cells) + r" \\")
    if sculpt:
        lines += [r"\midrule", "random selection of as many QCD jets & "
                  + " & ".join(_cell([1e2 * _sculpt_floor_distance(sculpt, e)], 1, fmt_pm, run=False) + " & --" for e in SCULPT_EFFS)
                  + (" & -- & --" if v3 else "") + r" \\"]
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", ""])


SCULPT_BOOT_WPS = (("eps_b=0.01", r"$\epsB = 1\%$"), ("eps_b=0.1", r"$\epsB = 10\%$"),
                   ("eps_s=0.5", r"$\epsS = 50\%$"))


# pairs the bootstrap file lacks that the sculpting figure draws (PI 2026-11-13, plan "tables"): their delta and run SD
# from the same distances (checked against the file's own pairs), no bootstrap SD, delta / sigma on the run SD alone
SCULPT_BOOT_EXTRA = (("L162", "L188"),)
NO_BOOT = r"$^{\dagger}$"


def _sculpt_distance(sculpt: dict, m: str, wp: str) -> float:
    """The Jensen-Shannon distance of run m at a working point of sculpting_boot.json ("eps_b=0.01", "eps_s=0.5")."""
    d = sculpt[m]
    return math.sqrt(d["working_points"][wp.split("=")[1]]["jsd"] if wp.startswith("eps_b") else d["sig_eff_50"]["jsd"])


def table_sculpt_boot(doc, runs, sculpt=None) -> str:
    """The sculpting differences of experiments/EVAL/sculpting_boot.py (head/sculpting_boot.json; plan 2026-11-08: a
    difference stands when |delta| > 3 sigma, sigma^2 = run SD^2 + bootstrap SD^2), one row group per working point:
    delta, run SD, bootstrap SD (distance x 10^2) and delta / sigma at two decimals; no bold (PI 2026-11-12). With
    sculpt (the per-run files), also SCULPT_BOOT_EXTRA, paired by seed, its sigma the run SD alone (NO_BOOT); every
    pair of the file is recomputed from sculpt first, and a mismatch is fatal. Rows in the tables' order (ALL_ROWS) of
    the network, then of its reference."""
    num = lambda v: f"{v:.2f}".replace("-", "$-$")      # noqa: E731
    arm = {m.removeprefix("mtx-").rsplit("-s", 1)[0]: a for a, ms in runs.items() for m in ms}
    word = {a: w for w, a in arm.items()}
    diffs = dict(doc["differences"])
    if sculpt:
        for name, d in diffs.items():
            for wp, _ in SCULPT_BOOT_WPS:
                pp = [_sculpt_distance(sculpt, x, wp) - _sculpt_distance(sculpt, y, wp) for x, y in d["pairs"]]
                if any(abs(u - v) > 1e-12 for u, v in zip(pp, d[wp]["distance"]["per_pair"])):
                    raise SystemExit(f"FATAL: sculpting_boot.json {name} {wp}: the paired distances differ from the "
                                     "sculpting files")
        for x, y in SCULPT_BOOT_EXTRA:
            pairs = [(m, r) for m in runs[x] for r in runs[y] if _idx(m) == _idx(r)]
            e = {"configuration": word[x], "reference": word[y], "pairs": pairs}
            for wp, _ in SCULPT_BOOT_WPS:
                pp = [_sculpt_distance(sculpt, m, wp) - _sculpt_distance(sculpt, r, wp) for m, r in pairs]
                delta, sd = float(np.mean(pp)), float(np.std(pp, ddof=1))
                e[wp] = {"distance": {"delta": delta, "run_sd": sd, "boot_sd": None, "z": delta / sd}}
            diffs[f"{word[x]}-{word[y]}"] = e
    order = lambda x: (ALL_ROWS.index(arm[x[1]["configuration"]]), ALL_ROWS.index(arm[x[1]["reference"]]))   # noqa: E731
    out = [r"\begin{tabular}{lcccc}", r"\toprule", r" & $\Delta$ & run SD & bootstrap SD & $\Delta/\sigma$ \\", r"\midrule"]
    for k, (wp, head) in enumerate(SCULPT_BOOT_WPS):
        out += ([r"\midrule"] if k else []) + [_full_row(5, head, italic=True)]
        for name, d in sorted(diffs.items(), key=order):
            part = arm[d["configuration"]] in FLAV + RAND
            if part and PARTITIONS not in out[-1] and not out[-1].startswith(r"\quad "):
                out.append(_full_row(5, PARTITIONS))
            x = d[wp]["distance"]
            if x["boot_sd"] is not None and (x["sigma"] is None or not math.isclose(x["sigma"], math.hypot(x["run_sd"], x["boot_sd"]), rel_tol=1e-9)
                                             or x["pass"] != (abs(x["delta"]) > 3 * x["sigma"])):
                raise SystemExit(f"FATAL: sculpting_boot.json {name} {wp}: sigma is not hypot(run SD, bootstrap SD) "
                                 "or pass is not |delta| > 3 sigma")
            boot = "--" if x["boot_sd"] is None else num(1e2 * x["boot_sd"])
            out.append((r"\quad " if part else "") + f"{LABEL[arm[d['configuration']]]} $-$ {LABEL[arm[d['reference']]]} & "
                       + " & ".join(num(1e2 * x[q]) for q in ("delta", "run_sd")) + f" & {boot} & " + num(x["z"])
                       + (NO_BOOT if x["boot_sd"] is None else "") + r" \\")
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


def table_coarse(co, runs, fmt_pm, rows) -> str:
    """Should-add 5 (PI 2026-11-07): linear probes of every network on the coarse targets of class-balanced
    JetClass-II at 1e6 (probe_bal.py --coarse): the 43- and 17-group levels of the hierarchy and the 17-class
    partition with no flavor boundary; accuracy and the macro AUC over the target's groups."""
    def val(m, ck, t, k):
        c = co.get(m, {}).get("cells", {}).get(ck, {}).get("jc2bal_coarse", {}).get(t, {}).get("1000000")
        return None if c is None else c[k]
    n = len(COARSE_TARGETS)
    out = [r"\begin{tabular}{l" + "cc" * n + "}", r"\toprule",
           " & " + " & ".join(r"\multicolumn{2}{c}{%s}" % w for _, w in COARSE_TARGETS) + r" \\",
           "".join(r"\cmidrule(lr){%d-%d}" % (2 + 2 * i, 3 + 2 * i) for i in range(n)),
           " & " + " & ".join(["accuracy & AUC"] * n) + r" \\", r"\midrule"]
    for arm in rows:
        ms, ck = ((INIT_MODELS, "init") if arm == INIT else (INIT_MODELS, "init_bn") if arm == INIT_BN
                  else (runs[arm], SOPHON_CK) if arm == SOPHON else (runs[arm], PRIMARY))
        cells = [_cell([val(m, ck, t, k) for m in ms], len(ms), fmt_pm, frac=True)
                 for t, _ in COARSE_TARGETS for k in ("accuracy", "macro_auc_ovr")]
        out += [r"\midrule"] * (arm == SOPHON) + [
            (LABEL[arm] if arm in (INIT, INIT_BN) else _row(arm, runs)) + " & " + " & ".join(cells) + r" \\"]
    return "\n".join(out + [r"\bottomrule", r"\end{tabular}", ""])


def qcdhf_fill(hf) -> dict:
    """The most charm and light QCD jets passing the R50 cut in any run of a pretrained network, for the caption."""
    n = {b: max((c["n_bkg_pass"] for d in hf.values() for ck, cell in d.get("cells", {}).items() if ck == PRIMARY
                 if (c := cell.get("background", {}).get(b, {}).get("natural", {}).get(QCDHF_WP))), default=0)
         for b in ("c", "light")}
    return {"__QCDHF_C__": str(n["c"]), "__QCDHF_LIGHT__": str(n["light"])}


def sculpt_fill(sculpt, runs) -> dict:
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
            "__SCULPT_CUT__": (r" or half of the resonant test jets, the \LitJetClassTwoNResonantClasses{} resonant "
                               r"classes taken together ($\epsS = 50\%$), where $R_{50} = 1/\epsB$, and $\epsS$ "
                               r"beside a distance is the fraction of the resonant test jets that pass" if v3 else ""),
            "__SCULPT_RUNS__": (PART_DEF if flav else "") + _runs("set", *(("flav",) if flav else ()), "mass")}


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
        sd = _level_sd(runs, f)
        for arm, ref in (("L162_MASS", "L162"), ("R16_Q1_MASS", "R16_Q1"), ("R16_Q1", "L188"), ("L162", "L188"),
                         ("R42_Q1", "L188"), ("R16_Q1", "R42_Q1")):
            r, k = _ratio_vals(runs, arm, ref, f)
            _emit_ratio(em, f"MlstSculpt{WORD[arm]}Over{WORD[ref]}{ew}", r, k, fmt_pm, index,
                        f"distance at QCD efficiency {e}: {arm} over {ref}, run by run", "distance", _two_run(runs, arm, ref),
                        ref_sd=sd)
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
    return tag, merge_budget(per, data / "budget" / "budget_metrics.json",
                             data / "budget" / "budget_long_metrics.json",
                             [data / "budget" / f for f in SCRATCH_FULL], data / "budget" / PRE_FULL), absent


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
    return (_cell([1.0], 1, fmt_pm, run=False) + (rf" \pending{{{have} of {n}}}" if have < n else "")) if have else r"\pending{no run yet}"


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
    tasks.insert(1, ("jc2", "accuracy", "accuracy", "JetClass-II"))      # class-balanced: the balanced accuracy
    tasks += list(FT_FULL_AUC)          # the AUC of the training-size figures' AUC versions (supplement)
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
            sd = _level_sd(runs, f)
            for arm, ref in FT_R30_CMP:
                r, k = _ratio_vals(runs, arm, ref, f)
                _emit_ratio(em, f"MlstFtCmp{_name_over(FT_WORD[key], arm, ref)}{sw}", r, k, fmt_pm, index,
                            f"fine-tuning {key} N={n}: (1 - {met}) of {arm} over {ref}, run by run", "errors",
                            _two_run(runs, arm, ref), ref_sd=sd)
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
            sd = _level_sd(runs, f)
            for arm, rf in FT_R30_CMP:
                r, k = _ratio_vals(runs, arm, rf, f)
                _emit_ratio(em, f"MlstFt{_name_over(tw + 'Rthirty', arm, rf)}{sw}", r, k, fmt_pm, index,
                            f"fine-tuning {key} N={n}: R30 of {arm} over {rf}, run by run", "rejection",
                            _two_run(runs, arm, rf), ref_sd=sd)
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


def ft_gpu_types(data: pathlib.Path) -> tuple[dict, dict]:
    """({init: sorted GPU names}, {init: fine-tuning cells whose read-out records no device}) of the
    fine-tuning cells (ft_mlst/manifests.json and the budget cells). budget_metrics.py records the
    device only for the top and quark/gluon cells (bench_metrics.cell_metrics), so the JetClass and
    class-balanced JetClass-II cells count as unrecorded until it records theirs."""
    out, nodev = collections.defaultdict(set), collections.Counter()
    f = data / "ft_mlst" / "manifests.json"
    if f.exists():
        for c in json.loads(f.read_text()).values():
            out[c["init"]].add(gpu_name(c["gpu_device_name"]))
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


def _rej_diff(pre, runs, arm, ref, c) -> tuple:
    """(paired ratios, is a difference, is a drop) of one class's fine-tuned rejection, arm over ref:
    a difference when the mean ratio departs from 1 by more than twice its SD and by more than its
    test-sample uncertainty (Poisson on the passing background jets, sqrt(1/n_a + 1/n_b), averaged over
    the pairs); a column where fewer than JC1_MIN_BKG background jets pass in any run is never one."""
    a = {_idx(m): _jc1_rej(pre, m, c) for m in runs[arm]}
    b = {_idx(m): _jc1_rej(pre, m, c) for m in runs[ref]}
    pairs = [(a[k], b[k]) for k in sorted(set(a) & set(b)) if a[k] and b[k]]
    r = [x["rejection"] / y["rejection"] for x, y in pairs]
    if len(r) < 3 or min(min(x["n_bkg_pass"], y["n_bkg_pass"]) for x, y in pairs) < V4.JC1_MIN_BKG:
        return r, False, False
    m, sd = float(np.mean(r)), float(np.std(r, ddof=1))
    test = m * float(np.mean([math.sqrt(1 / x["n_bkg_pass"] + 1 / y["n_bkg_pass"]) for x, y in pairs]))
    d = abs(m - 1) > 2 * sd and abs(m - 1) > test
    return r, d, d and m < 1


def jetclass_macros(em, pre: dict, runs: dict, src, fmt_pm, words) -> None:
    """MlstFtJcOneRej<class><arm>Over<ref>: the fine-tuned rejection of a JetClass signal class at
    ParT's working point, run by run; ...RandOverFull over the partitions (and its range);
    MlstFtJcOne<arm>NRejDrop / NRejDiff / RejDropClasses: the classes whose ratio to the 188-class
    model meets the difference rule (_rej_diff); MlstFtJcOneAccRandMinusCoarse."""
    heads = dict(V4.JC1_COLS)
    em.macro("MlstFtJcOneNSignal", words(len(V4.JC1_COLS)), src, "JC1_COLS (count)", "JetClass signal classes")
    for c, _ in V4.JC1_COLS:
        cw = JC1_WORD[c]
        f = lambda m, c=c: None if (x := _jc1_rej(pre, m, c)) is None else x["rejection"]
        sd = _level_sd(runs, f)
        for arm, ref in FT_R30_CMP:
            r, k = _ratio_vals(runs, arm, ref, f)
            _emit_ratio(em, f"MlstFtJcOneRej{cw}{WORD[arm]}Over{'Full' if ref == 'L188' else WORD[ref]}", r, k, fmt_pm,
                        src, f"cells.<run>.N1000000.s1.jc1.per_class.{c}: rejection of {arm} over {ref}, run by run",
                        "rejection", _two_run(runs, arm, ref), ref_sd=sd)
        r = _rand_over(runs, "L188", f)
        _emit_ratio(em, f"MlstFtJcOneRej{cw}RandOverFull", r, len(r), fmt_pm, src,
                    f"per_class.{c}: rejection of each random partition over the 188-class run of its seed", "rejection")
        v = [x for x in r if x is not None]
        em.macro(f"MlstFtJcOneRej{cw}RandOverFullRange", f"{_sig2(min(v))}--{_sig2(max(v))}" if v else r"\pending{runs}",
                 src, f"per_class.{c}: smallest and largest of the partitions' ratios")
    for arm in ("L162", "R42_Q1", "R16_Q1"):
        res = {c: _rej_diff(pre, runs, arm, "L188", c) for c, _ in V4.JC1_COLS}
        drop = [c for c, (_, _, dr) in res.items() if dr]
        diff = [c for c, (_, d, _) in res.items() if d]
        w = WORD[arm]
        note = ("classes whose paired rejection ratio to the 188-class model departs from 1 by more than twice its SD "
                "and its counting uncertainty, columns with fewer than JC1_MIN_BKG passing jets excluded")
        em.macro(f"MlstFtJcOne{w}NRejDrop", words(len(drop)), src, "per_class rejections", note + ", ratio below 1 (count)")
        em.macro(f"MlstFtJcOne{w}NRejDiff", words(len(diff)), src, "per_class rejections", note + " (count)")
        names = [heads[c] for c in drop]
        em.macro(f"MlstFtJcOne{w}RejDropClasses", ("none" if not names else names[0] if len(names) == 1 else
                                                    ", ".join(names[:-1]) + " and " + names[-1]), src,
                 "per_class rejections", "the classes of ...NRejDrop, in ParT's column order")
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
    for arm in rows:
        recs = [_probe_rec(fits, m, "jc1") for m in runs[arm]]
        cells = [_cell([None if r is None else r["accuracy"] for r in recs], len(recs), fmt_pm, frac=True),
                 _cell([None if r is None else r["macro_auc_ovr"] for r in recs], len(recs), fmt_pm, frac=True)]
        for c, _ in V4.JC1_COLS:
            w = [None if r is None else r["rejection_vs_qcd"].get(str(idx[c])) for r in recs]
            mark = r"$^\dagger$" if any(x is not None and x["n_bkg_pass"] < V4.JC1_MIN_BKG for x in w) else ""
            cells.append(_cell([None if x is None else (Bound(x["rejection"]) if x.get("is_bound") else x["rejection"])
                                for x in w], len(recs), fmt_pm) + mark)
        out += [r"\midrule"] * (arm == SOPHON) + [_row(arm, runs) + " & " + " & ".join(cells) + r" \\"]
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


def table_jc1_class_auc(ft_docs: dict, probe_docs: dict, runs: dict, fmt_pm, arms=ALL_ROWS, init=INIT) -> str:
    """Each JetClass class against the other nine, at 10^6 jets (curves_v2/jc1_ft_ovr, jc1_probe_ovr; the points of
    the JetClass per-class figures): arms (the paper: the random partitions pooled, runs from _pooled) fine-tuned with
    the network from scratch, and probed with the random network init."""
    scratch = [m for m in sorted(ft_docs) if m.startswith("scratch")]

    def rows(docs: dict, extra: list) -> list:
        out = []
        for name, ms in [(_row(a, runs), runs[a]) for a in arms] + extra:
            have = [m for m in ms if m in docs]
            if have:
                out.append(name + " & " + " & ".join(
                    _cell([docs[m]["curves"][c]["auc"] for m in have], len(ms), fmt_pm, frac=True) for c, _ in JC1_AUC_COLS) + r" \\")
        return out

    head = [r"\begin{tabular}{l" + "c" * len(JC1_AUC_COLS) + "}", r"\toprule",
            " & " + " & ".join(t for _, t in JC1_AUC_COLS) + r" \\", r"\midrule"]
    part = lambda t: [r"\multicolumn{" + str(1 + len(JC1_AUC_COLS)) + r"}{l}{\textit{" + t + r"}} \\"]   # noqa: E731
    return "\n".join(head + rows(ft_docs, [(SCRATCH_LABEL, scratch)])
                     + [r"\bottomrule", r"\end{tabular}", "%SPLIT%"] + head
                     + rows(probe_docs, [(_row(init, runs), runs[init])]) + [r"\bottomrule", r"\end{tabular}", ""])


def per_class_csv(fits, runs, root) -> str:
    """The data release's per-class table of JetClass-II (class-balanced test set, largest training set): the
    linear-probe AUC of each class against the rest, mean over the runs of each configuration."""
    arms = MAIN + MASS + RAND + FLAV
    per = {a: [p for m in runs[a] if (p := _per_class(fits, m))] for a in arms}
    lines = ["jet_label,class_name," + ",".join(arms)]
    for r in _class_rows(root):
        c = int(r["jet_label"])
        vals = [float(np.mean([p[c] for p in per[a] if c in p])) if any(c in p for p in per[a]) else "" for a in arms]
        lines.append(f"{c},{r['class_name']}," + ",".join("" if v == "" else f"{v:.5f}" for v in vals))
    return "\n".join(lines) + "\n"




def per_class_auc(fits, per, runs, arm: str, mode: str) -> dict:
    """{class: [AUC of the class against the rest, one a run of arm]} on JetClass-II, as the per-class figures read
    them (gallery_perclass.SOURCES["jc2"]): the linear probe on one million class-balanced jets (_per_class), or the
    network fine-tuned on as many (mode "ft", the per_class_auc of the fine-tuning read-out)."""
    out = collections.defaultdict(list)
    for m in runs[arm]:
        if mode == "lp":
            p = _per_class(fits, m)
        else:
            p = _ft(per, "jc2", m, FT_RATIO_N, "per_class_auc") if per.get("jc2") else None
            p = None if p is None else {int(c): v for c, v in p.items() if v is not None}
        for c, v in (p or {}).items():
            out[c].append(v)
    return out


def table_per_class(fits, per, runs, fmt_pm, root, native_tex) -> str:
    """tab:perclass (supplement): per class of JetClass-II, in the order of the per-class figures' grouping (by the
    17-class group), the AUC of the class against the rest for the label sets and the 17-class partitions (the random
    partitions pooled, runs from _pooled), as a linear probe and fine-tuned: every point of the JetClass-II per-class figures.
    Two turned longtables under one number, the label sets and then the 17-class partitions (ten columns of five
    decimals do not fit one turned page), each with its column names on every page and the two modes as blocks; each
    block's columns at their own precision."""
    classes = _class_rows(root)
    out = [r"\begin{landscape}", "{\\scriptsize", r"\setlength{\tabcolsep}{2pt}", LT_SETUP]
    for g, (group, cols) in enumerate(((None, SETS), (PARTITIONS, FLAV + (RAND_POOLED,)))):
        head = ([rf" & \multicolumn{{{len(cols)}}}{{c}}{{{group}}} \\", rf"\cmidrule(lr){{2-{1 + len(cols)}}}"] if group else []) + [
            "class & " + " & ".join(LABEL[a] for a in cols) + r" \\"]
        more = r"\multicolumn{" + str(1 + len(cols)) + r"}{@{}l}{\textit{Table~\ref{tab:perclass} (continued)}} \\"
        out += ([r"\addtocounter{table}{-1}"] if g else []) + [
            r"\begin{longtable}{@{\extracolsep{\fill}}l" + "c" * len(cols) + "@{}}",
            r"\caption{__CAPTION__}\label{tab:perclass}\\" if not g else more, r"\toprule", *head, r"\midrule", r"\endfirsthead",
            more, r"\toprule", *head, r"\midrule", r"\endhead", r"\bottomrule", r"\endlastfoot"]
        for k, mode in enumerate(("lp", "ft")):
            vals = {a: per_class_auc(fits, per, runs, a, mode) for a in cols}
            rows, last = [], None
            for r in classes:
                c = int(r["jet_label"])
                if last is not None and r["R16_Q1"] != last:
                    rows.append(r"\midrule")
                last = r["R16_Q1"]
                rows.append(native_tex(r["class_name"]) + " & " + " & ".join(
                    _cell(vals[a][c], len(runs[a]), fmt_pm, frac=True) if vals[a].get(c) else "--" for a in cols) + r" \\")
            out += ([r"\midrule[\heavyrulewidth]"] if k else []) + [
                rf"\multicolumn{{{1 + len(cols)}}}{{@{{}}l}}{{\textit{{{HEADING[mode]}}}}} \\*"] + uniform_columns("\n".join(rows)).split("\n")
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


def table_boot(boot: dict, runs: dict, fits=None, per=None, balanced: bool = True) -> tuple[str, dict]:
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
                    get = lambda m, src=src: None if fits is None else _probe_rec(fits, m, src)
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
PARTREPRO = (("part", "ParT's training script, unchanged"), ("head", "ParT's script with Sophon's output head"),
             ("rescale", r"ParT's script with Sophon's jet-$p_{\mathrm{T}}$ rescaling"))


def partrepro(em, data: pathlib.Path, per, fmt_pm, missing: list, runs: dict | None = None) -> str | None:
    """A9 and its macros: ParT's top-tagging script on our cluster, unchanged, with Sophon's output head, and with
    Sophon's rescaling of the four-momenta: each of the two a single change to ParT's script, not cumulative
    (scripts/build_partrepro_jobs.py VARIANTS), one run each. ParT's published row and our network trained from
    scratch are in the comparison table only (tab:published; PI 2026-11-12: one table per number)."""
    docs = {v: json.loads(p.read_text()) for v, _ in PARTREPRO if (p := data / "partrepro" / f"{v}.json").exists()}
    for v, _ in PARTREPRO:
        d = docs.get(v)
        src = data / "partrepro" / f"{v}.json" if d else data / "index.json"
        em.macro(f"MlstPartreproRthirty{v.title()}", _sep(str(round(d["r30"]))) if d else r"\pending{reproduction}", src,
                 "r30", f"ParT reproduction on top, {v}: R30 (one run)")
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
    for v, name in PARTREPRO:       # one run each: marked
        d = docs.get(v)
        out.append(name + " & " + (" & ".join([_at(d[k], place[k]) + SINGLE for k in keys]
                                              + [_sep(str(d['r30_n_bkg_pass'])) + SINGLE])
                                   if d else r"\multicolumn{5}{c}{\pending{run}}") + r" \\")
    return "\n".join(out + [r"\bottomrule", r"\end{tabular}", ""])


def table_lrsweep(doc: dict) -> str:
    """A4: the learning-rate sweep at 10^4 top and quark/gluon jets: each rate's best validation
    accuracy per task and their sum; a star on the rates chosen (mlst_v4.choose_rates), on the sum
    for the pretrained weights and on each task's accuracy from scratch."""
    pre, scr = V4.choose_rates(doc)
    by = collections.defaultdict(dict)
    for key, m in doc["cells"].items():
        tree, leaf = key.split("/")[:2]
        init, lr = leaf.rsplit("_lr", 1)
        by[(SCRATCH_LABEL if init == "scratch" else LABEL["L188"], lr)][tree] = m["early_stop"]["best_metric"]
    out = [r"\begin{tabular}{llccc}", r"\toprule", r" & learning rate & top & quark/gluon & sum \\", r"\midrule"]
    star = r"$^{\ast}$"
    for (kind, lr), v in sorted(by.items(), key=lambda x: (x[0][0] == SCRATCH_LABEL, float(x[0][1]))):
        vals = [v.get("leg_top"), v.get("leg_qg")]
        cells = ["--" if x is None else f"{x:.4f}" + SINGLE + (star if kind == SCRATCH_LABEL and scr[t] == lr else "")
                 for t, x in zip(("leg_top", "leg_qg"), vals)]
        s = f"{sum(vals):.4f}" + SINGLE if None not in vals else "--"
        out.append(f"{kind} & {_power(lr)} & " + " & ".join(cells) + f" & {s}" + (star if kind != SCRATCH_LABEL and lr == pre else "") + r" \\")
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
    if isinstance(r, list):
        v = [x.get(key) for x in r]
        return None if any(x is None for x in v) else float(np.mean(v))
    return r.get(key)


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
# the frozen features, the linear probe, the direct discriminant); the X->bs closest-class output, which no sentence
# cites, is the supplement's tab:xbxnode (L13)
XBX_REJ_PAGES = (("xbb", ("head", "linear", "zero_shot_xbb")), ("xbs", ("head", "linear")))
XBX_NODE_PAGES = (("xbs", ("zero_shot_xbq",)),)
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


def xbx_section(em, root, data, index, runs, missing, out, fmt_pm, fmt_int) -> tuple:
    """X->bb and X->bs from the pulled fits through experiments/XBS/xbx_report.py, and the rejection
    ratios and table of A15; red markers until they exist. Returns (the xbx_report module, the fits)."""
    from types import SimpleNamespace
    spec = importlib.util.spec_from_file_location("xbx_report", root / "experiments/XBS/xbx_report.py")
    XR = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(XR)
    xfits = XR.load(data / "xbx") if (data / "xbx").is_dir() else {}
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
        out["tables/mlst_xbx_node.tex"] = _float("xbx_node", table_xbx_rej(xfits, scr, runs, TAB, XBX_NODE_PAGES))
        xbx_ratio_macros(em, xfits, scr, runs, index, fmt_pm)
    else:
        missing.append("X->bb and X->bs fits -- experiments/FIGS/data/v2/xbx/")
    if not scr:
        missing.append("X->bs from scratch -- experiments/FIGS/data/v2/xbx_scratch/")
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
SUMMARY_HEAD = {"accuracy": "accuracy", "r30": "$R_{30}$", "macro_auc_ovr": "AUC", "auc": "AUC"}
SUMMARY_CONTROLS = FLAV + (RAND_POOLED,)
SUMMARY_NO_SOPHON = ("top", "qg", "qg_herwig")      # its probes there have the missing inputs at zero: not drawn


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
    fine-tuning at n."""
    if not per.get(key) or n not in _sizes(per, key):
        return None
    return lambda m: _ft(per, key, m, n, met)


def summary_scratch(per, key: str, n: int, met: str) -> list:
    """Each from-scratch seed's value on n training jets, as Fig. 2 and its full version draw them
    (gallery_labelset._scratch_value)."""
    return [r.get(met) for r in per.get(key, {}).get(SCRATCH, {}).get(n, []) if r.get(met) is not None]


def table_summary(fits, fill, chosen, per, runs, fmt_pm, grid=SUMMARY_GRID, key="summary") -> str:
    """The values of the headline family on grid, one %SPLIT% part (a turned page) per mode: linear probes (the label
    sets, each mass network under its own, the 17-class partitions with the five random partitions as one row, the
    random network, and below a rule the released Sophon checkpoint on JetClass-II and JetClass), then fine-tuned (the
    same networks, from scratch last). fits: the probe fits with the released checkpoint's merged in, as the figures
    read them; runs: with SOPHON. A row with no value for a task: a dash. A partition row with no value on any task is
    left out. key: the table's BOLD entry (tab:summary, tab:summaryauc)."""
    runs = _pooled(runs)

    def cells(arm, get):
        ms = runs[arm]
        out = []
        for g, (_, _, ds, met) in zip(get, grid):
            v = [g(m) for m in ms] if g and not (arm == SOPHON and ds in SUMMARY_NO_SOPHON) else []
            out.append(_cell(v, len(ms), fmt_pm, frac=met != "r30") if any(x is not None for x in v) else "--")
        return out

    def scratch():
        out = []
        for k, n, _, met in grid:
            v = summary_scratch(per, k, n, met)
            out.append(_cell(v, len(v), fmt_pm, frac=met != "r30") if v else "--")
        return out

    row = lambda name, c: f"{name} & " + " & ".join(c) + r" \\"       # noqa: E731
    nc = 1 + len(grid)
    # two header rows: the task (quark/gluon over its two columns), then the metric and, under quark/gluon, the
    # generator of the jets scored (both trained on Pythia jets)
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
        body = [_full_row(nc, HEADING[mode], italic=True)] + [row(LABEL[a], cells(a, get)) for a in SETS]
        ctl = [(r"\quad " + LABEL[a], cells(a, get)) for a in SUMMARY_CONTROLS]
        ctl = [(n, c) for n, c in ctl if any(x != "--" for x in c)]
        if ctl:
            body += [_full_row(nc, PARTITIONS)] + [row(n, c) for n, c in ctl]
        if mode == "lp":
            body.append(row(LABEL[INIT], cells(INIT, get)))
            c = cells(SOPHON, get) if SOPHON in runs else []
            if any(x != "--" for x in c):
                body += [r"\midrule", row(LABEL[SOPHON], c)]
        else:
            body.append(row(SCRATCH_LABEL, scratch()))
        # each part at its own precision per column (uniform_columns), so a fine-tuned value prints at its own place
        parts.append("\n".join(head + uniform_columns(_bold(key, "\n".join(body))).split("\n")
                               + [r"\bottomrule", r"\end{tabular}", ""]))
    return "%SPLIT%".join(parts)


# ------------------------------------------------------------------ published models (main text), the other label sets (appendix)
# PI 2026-11-12: one main-text table beside the published models, with our network from scratch and fine-tuned from
# 188 classes only; every other label set's fine-tuned row in one appendix table. Apart from tab:summary a number
# has one table: the 188-class and from-scratch rows of JetClass, top and quark/gluon (Pythia) are in tab:published
# and not in tab:labelsets.
THIS_WORK = " (this work)"
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
                + uniform_columns(_bold(key, "\n".join(blocks[t]))).split("\n")]
        out.append("\n".join(head + body + [r"\bottomrule", r"\end{tabular}", ""]))
    return "%SPLIT%".join(out)


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
    for prefix, cells, _ in lit + sorted(ours, key=lambda r: not r[0].startswith(SCRATCH_LABEL)):
        name, jets, params = V4._depth0(prefix)
        name = {LABEL["L188"]: ft, SCRATCH_LABEL: SCRATCH_LABEL + THIS_WORK}.get(name, name)
        rows.append(f"{name} & {params} & " + jets.replace(" M", r"\,M") + f" & {cells[0]} & {cells[1]} & -- & -- \\\\")
    blocks["jc1"] = rows
    for task in ("top", "qg"):
        lit, ours = V4.bench_rows(task, per, runs, ("L188",), LABEL, SCRATCH, fmt_pm, fmt_pm.two,
                                  kin=(LABEL["KIN"], KIN_INIT) if task == "top" else None, published=True)
        rows = [f"{name} & {params} & " + ("--" if kind == "observable" else _jets(FULL_N[task])) + " & "
                + " & ".join(cells) + r" \\" for name, cells, params, kind in lit]
        by = {name: (cells, params) for name, cells, params in ours}
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
    refs = []
    if kin and _probe_rec(fits, KIN_MODEL, "top") is not None:
        refs.append((LABEL["KIN"], cells([KIN_MODEL])[:len(BENCH2)] + ["--"] * len(BENCH2) * (len(BENCH_GROUPS) - 1)))
    refs += [(LABEL[a], cells(runs[a])) for a in rows if a in (INIT, INIT_BN)]
    if SOPHON in rows:
        refs += [r"\midrule", (LABEL[SOPHON], cells(runs[SOPHON]))]
    return _bench_groups("probes", "lp", nets, refs)


def table_labelsets(per: dict, runs: dict, fmt_pm, rows=None) -> str:
    """tab:labelsets: the fine-tuned networks on the full top and quark/gluon training sets, read as Table 1 reads
    them (summary_ft, summary_scratch), the accuracy and R50 on top and quark/gluon (Pythia, Herwig); rows (default
    SETS, the 17-class partitions, the five random partitions pooled), then from scratch."""
    pooled = _pooled(runs)
    rows = rows or SETS + FLAV + (RAND_POOLED,)

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
    return _bench_groups("labelsets", "ft", nets, [(SCRATCH_LABEL, cells(None))])


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
        blocks, shown = [], None
        for t in tasks:
            _, ds, met, title = next(x for x in SIZE_TASKS if x[0] == t)
            rs = {a: series(t, ds, met, a) for a in rows + (ref[1],)}
            sizes = sorted({n for v in rs.values() for n in v if n != largest[t]})
            if shown not in (None, sizes):
                raise SystemExit(f"FATAL: {key}: the tasks {tasks} have different training sizes")
            shown = sizes
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
                      "qg_herwig": "quark/gluon tagging, scored on Herwig jets",
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
        for name, word in (("head", "frozen-feature classifier"), ("linear", "linear probe"),
                           ("zero_shot_xbb", "direct discriminant")):
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


# ------------------------------------------------------------------ compute (appendix, Data availability)
def table_compute(comp, runs, fmt_int, extra=(), ft_gpus=None, ft_nodev=None) -> tuple[str, dict]:
    """Per pretraining run: hours on each GPU type (completed epochs, training and validation passes,
    as each epoch recorded its device), their sum, and the GPU types its fine-tuning cells ran on
    (their manifests record the device, not the hours), with a marker counting its cells whose
    read-out records no device (ft_nodev). extra: (label, model) rows beside the grid."""
    # a longtable in landscape whose two lists of GPU types wrap inside their columns and which breaks across
    # pages (PI 2026-11-11: it was one wide tabular shrunk to fit); the caption is the placeholder __CAPTION__
    head = r"network & GPU types (hours) & GPU-hours & fine-tuning GPU types \\"
    wrap = lambda cm: r">{\raggedright\arraybackslash}p{" + cm + "cm}"      # noqa: E731
    lines = [r"\begin{landscape}", "{\\small", LT_SETUP,
             r"\begin{longtable}{@{\extracolsep{\fill}}l" + wrap("6.0") + "r" + wrap("9.6") + "@{}}",
             r"\caption{__CAPTION__}\label{tab:compute}\\", r"\toprule", head, r"\midrule", r"\endfirsthead",
             r"\multicolumn{4}{@{}l}{\textit{Table~\ref{tab:compute} (continued)}} \\", r"\toprule", head, r"\midrule", r"\endhead",
             r"\bottomrule", r"\endlastfoot"]
    hours = {}
    arms = [a for a in ALL_ROWS if a in runs] + [a for a in runs if a not in ALL_ROWS + (INIT, INIT_BN)]   # L5 order
    items = [(arm, f"{LABEL[arm]}, seed {_idx(m)}", m) for arm in arms for m in runs[arm]]
    items += [(None, lab, m) for lab, m in extra]
    for arm, name, m in items:
        ftg = ", ".join(map(_whole, (ft_gpus or {}).get(m[len("mtx-"):], []))) or "--"
        if (k := (ft_nodev or {}).get(m[len("mtx-"):])):
            ftg += rf" \pending{{{k} cells without a device}}"
        if m not in comp:
            lines.append(rf"{name} & \pending{{training}} & & {ftg} \\")
            continue
        h = collections.Counter()
        for e in comp[m]["epochs"]:
            h[gpu_name(e["device"])] += (e["train_seconds"] + e["val_seconds"]) / 3600
        if arm is not None:
            hours.setdefault(arm, collections.Counter()).update(h)
        types = ", ".join(_whole(f"{g} ({h[g]:.0f})") for g in sorted(h, key=lambda g: -h[g]))
        lines.append(rf"{name} & {types} & {fmt_int(round(sum(h.values())))} & {ftg} \\")
    return "\n".join(lines + [r"\end{longtable}", "}", r"\end{landscape}", ""]), hours


def compute_macros(em, comp, runs, hours, index, fmt_int) -> None:
    total = collections.Counter()
    for arm in MAIN:
        total.update(hours.get(arm, {}))
    top = max(total, key=total.get) if total else None
    shares = [hours[a][top] / sum(hours[a].values()) for a in MAIN if a in hours] if top else []
    paper = round(sum(h for arm in hours.values() for h in arm.values()))
    em.macro("ComputeTotalGpuHours", fmt_int(paper), index, "GPU-hours of every reported pretraining run, training and "
             "validation passes")
    allh = paper + (round(sum((e["train_seconds"] + e["val_seconds"]) / 3600 for e in comp[KIN_MODEL]["epochs"]))
                    if KIN_MODEL in comp else 0)
    em.macro("MlstComputeTotalGpuHours", fmt_int(allh), index,
             "GPU-hours of every pretraining run the paper reports, the kinematic-only model included")
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


# ------------------------------------------------------------------ per-run values (A1)
def table_per_run(fits, per, runs, fmt_pm, fill=None, chosen=None) -> str:
    """A1: every pretraining run's own values at the largest training set: linear probe (JetClass-II
    AUC, JetClass accuracy, R30 on top and quark/gluon under the substitution chosen on validation, zero
    while it is pending) and fine-tuning (JetClass-II AUC, JetClass accuracy, R30 on top,
    quark/gluon and Herwig)."""
    fill, chosen = fill or {}, chosen or {}
    # three short header lines and a wrapping run column keep the table inside the text block, upright
    head = (r" & \multicolumn{4}{c}{linear probe on frozen features} & \multicolumn{5}{c}{Sophon's network, fine-tuned} \\" "\n"
            r"\cmidrule(lr){2-5}\cmidrule(lr){6-10}" "\n"
            r" & JetClass-II & JetClass & top & quark/ & JetClass-II & JetClass & top & quark/ & Herwig \\" "\n"
            r" & AUC & accuracy & $R_{30}$ & gluon $R_{30}$ & AUC & accuracy & $R_{30}$ & gluon $R_{30}$ & $R_{30}$ \\")
    one = lambda x: "--" if x is None else f"{x:.4f}" if abs(x) < 1 else _tab1(x)   # accuracy, AUC: four decimals

    def ft(key, m, met):
        if not per or not _sizes(per, key):
            return None
        return _ftv(per, key, m, _sizes(per, key)[-1], met)
    rows = []
    items = [(f"{LABEL[a]}, run {_idx(m)}", m) for a in ALL_ROWS for m in runs[a]] + [(LABEL["KIN"], KIN_MODEL)]
    for name, m in items:
        p2, p1 = _probe_rec(fits, m, "jc2_188", PAIR_N), _probe_rec(fits, m, "jc1")
        pt, pq = _bench_rec(fits, fill, chosen, m, "top"), _bench_rec(fits, fill, chosen, m, "qg")
        cells = [one(None if p2 is None else p2["macro_auc_ovr"]), one(None if p1 is None else p1["accuracy"]),
                 one(_rej(pt, "r30")), one(_rej(pq, "r30")), one(ft("jc2", m, "macro_auc_ovr")),
                 one(ft("jc1", m, "accuracy")), one(ft("top", m, "r30")), one(ft("qg", m, "r30")),
                 one(ft("qg_herwig", m, "r30"))]
        rows.append(name + " & " + " & ".join(cells) + r" \\")
    if per:
        seeds = max((len(per[k].get(SCRATCH, {}).get(_sizes(per, k)[-1], [])) for k in ("jc1", "top", "qg")
                     if _sizes(per, k)), default=0)
        for s in range(seeds):
            vals = []
            for key, met in (("jc2", "macro_auc_ovr"), ("jc1", "accuracy"), ("top", "r30"), ("qg", "r30"), ("qg_herwig", "r30")):
                v = _scratch(per, key, _sizes(per, key)[-1], met) if _sizes(per, key) else []
                vals.append(one(v[s] if s < len(v) else None))
            rows.append(f"{SCRATCH_LABEL}, run {s + 1} & -- & -- & -- & -- & " + " & ".join(vals) + r" \\")
    # in landscape across the whole line at the tables' \\small, the row labels on one line (PI 2026-11-11)
    return "\n".join([r"\begin{landscape}", "{\\small", LT_SETUP,
                      r"\begin{longtable}{@{\extracolsep{\fill}}l" + "c" * 9 + "@{}}",
                      r"\caption{__CAPTION__}\label{tab:perrun}\\", r"\toprule", head, r"\midrule", r"\endfirsthead",
                      r"\multicolumn{10}{@{}l}{\textit{Table~\ref{tab:perrun} (continued)}} \\", r"\toprule", head, r"\midrule", r"\endhead",
                      r"\bottomrule", r"\endlastfoot", *rows, r"\end{longtable}", "}", r"\end{landscape}", ""])


# ------------------------------------------------------------------ captions and assembly
SPREAD = ("Mean $\\pm$ SD over the \\MlstNRunsWord{} pretraining runs of each level of the class "
          "hierarchy; configurations with \\RandFlavRuns{} runs: over those. Every random partition has 17 "
          "classes; each numbered one (1--5) has a single run, numbered by the pretraining seed it shares with the "
          "\\Kfull-class run of that seed. The random network without pretraining: three initializations.")
SPREAD3 = "Mean $\\pm$ SD over the \\MlstNRunsWord{} pretraining runs."
# the paper's tables (PI 2026-11-12): the five random partitions as one row
POOLED = "Random partitions: mean $\\pm$ SD over the five partitions into 17 classes, one run each."
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
# the pairs caption's note where a 188-class run separates a pair's test jets completely (tab:pairs '--')
ZERO_NOTE = (" A dash in a ratio column: the \\Kfull-class run of that seed separates the test jets almost "
             "completely ($1-\\mathrm{AUC} < 10^{-4}$), so a ratio to it would mean nothing.")


# the reference row of the frozen-probe tables; the differences are the verified ones (PI 2026-11-08): the stream and its
# loader (pretrain_v2.py: a fresh seeded stream every epoch), the initialization, the checkpoint rule (best_window_epoch)
SOPHON_NOTE = ("Below the rule, the released Sophon checkpoint~\\cite{sophon}, probed in the same way, lies outside the "
               "controlled comparison: it was pretrained on another stream of training jets, read by weaver's loader (our "
               "runs draw a fresh seeded stream every epoch), from its own initialization, and selected by its own "
               "checkpoint rule (ours: the best validation epoch of the last ten).")
SOPHON_SHORT = "Below the rule, the released Sophon checkpoint, outside the controlled comparison (table~\\ref{tab:probes})."
SOPHON_ZERO = (" Its top and quark/gluon probes set the missing inputs to zero, where our rows use the substitution "
               "chosen on validation.")


def zero_note(fits, runs, ft=None) -> str:
    """The tab:pairs caption's ZERO_NOTE when a 188-class run, probed or fine-tuned (ft, load_pairs_ft), separates
    some pair's test jets completely."""
    get = lambda m, k: ((ft or {}).get(m, {}).get(k) or {}).get("auc")      # noqa: E731
    return ZERO_NOTE if any(separated(fits, runs, k) or separated(fits, runs, k, get) for k, *_ in PAIRS) else ""


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
               + " " + KIN_NOTE + " " + SOPHON_NOTE + SOPHON_ZERO + " " + AUC_DEF + " $^{*}$: a probe that did not converge."),
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
               + " " + KIN_NOTE + " " + SOPHON_NOTE + SOPHON_ZERO + " " + AUC_DEF + " $^{*}$: a probe that did not converge."),
    "probes_n": ("tab:probesn", "The training-size points of the linear probes below each task's largest training set."),
    "ft": ("tab:ft", "The training-size points of the fine-tuned networks below each task's full training set."),
    "probes_n_full": ("tab:probesnfull", "Linear probes on the frozen features of networks pretrained on JetClass-II with 188, 162, 43 and 17 classes and of the controls, trained on "
                 "each task with the number of jets in the column headers: class-balanced "
                 "JetClass-II AUC averaged over the classes, JetClass accuracy, and $R_{30}$ on top, quark/gluon and Herwig jets with "
                 "the missing inputs substituted as chosen on validation (\\LpSubstitutionTop{} for top, "
                 "\\LpSubstitutionQg{} for quark/gluon). " + SPREAD + " Here the \\RandVtwoNPartitions{} random "
                 "partitions are pooled into one row, mean $\\pm$ SD over their runs. " + KIN_NOTE
                 + " $^{*}$: a probe that did not converge."),
    "pairs": ("tab:pairs", "Pairs of JetClass-II classes separated by networks pretrained on JetClass-II with 188, 162, 43 and 17 classes and by the controls: "
              "a binary linear probe on the frozen features, trained on up to \\JcBalPairTrainPerSide{} jets of each "
              "side from held-out JetClass-II files and scored on up to \\JcBalPairTestPerSide{} of each side from the "
              "JetClass-II test files, each side in its natural class mix; "
              "the network fine-tuned on $10^{6}$ class-balanced JetClass-II jets with the \\Kfull{} classes, "
              "scored on the pair's jets among \\LpNTestJcii{} JetClass-II test jets at their natural composition by the "
              "summed output scores of the first side over those of both sides. The two modes are scored on different "
              "test jets, so they are compared within a mode only. "
              "The AUC. "
              "No pretraining: fine-tuned from scratch, three seeds, no probe. $^{\\dagger}$: the label set merges the two "
              "sides into one class; $^{\\ddagger}$: it merges the first side with part of the second; $^{\\parallel}$: some of the five random partitions merge the two sides. " + SPREAD_POOLED
              + " The $b$-quark boundary and the random flavor boundary lie only in the four-quark decays; every label "
              "set but the \\Kfull-class one merges all QCD classes. The random network without pretraining, as initialized and "
              "with its BatchNorm statistics recomputed, has no label set, and no fine-tuning. The $X\\to ee$ vs $\\mu\\mu$ probe is trained on the "
              "jets of the two classes in the class-balanced training set (\\MlstPairEemmNTrain{} jets) and scored on every "
              "JetClass-II test jet of the two classes (\\MlstPairEemmNTestPerSide{} per side). A mark applies to the "
              "configuration.__ZERO__"),
    "pairs_full": ("tab:pairsfull", "Pairs of JetClass-II classes separated by networks pretrained on JetClass-II with 188, 162, 43 and 17 classes and by the controls, every random partition: "
              "a binary linear probe on the frozen features, trained on up to \\JcBalPairTrainPerSide{} jets of each "
              "side from held-out JetClass-II files and scored on up to \\JcBalPairTestPerSide{} of each side from the "
              "JetClass-II test files, each side in its natural class mix; "
              "the network fine-tuned on $10^{6}$ class-balanced JetClass-II jets with the \\Kfull{} classes, "
              "scored on the pair's jets among \\LpNTestJcii{} JetClass-II test jets at their natural composition by the "
              "summed output scores of the first side over those of both sides. The two modes are scored on different "
              "test jets, so they are compared within a mode only. "
              "The AUC and the ratio $\\rho$ "
              "of $1-\\mathrm{AUC}$, equation~(\\ref{eq:ratio}), to the \\Kfull-class run of the same seed in the same mode. "
              "No pretraining: fine-tuned from scratch, three seeds, no probe. $^{\\dagger}$: the label set merges the two "
              "sides into one class; $^{\\ddagger}$: it merges the first side with part of the second. " + SPREAD
              + " The $b$-quark boundary and the random flavor boundary lie only in the four-quark decays; every label "
              "set but the \\Kfull-class one merges all QCD classes. The random network without pretraining, as initialized and "
              "with its BatchNorm statistics recomputed, has no label set, no $\\rho$ and no fine-tuning. The $X\\to ee$ vs $\\mu\\mu$ probe is trained on the "
              "jets of the two classes in the class-balanced training set (\\MlstPairEemmNTrain{} jets) and scored on every "
              "JetClass-II test jet of the two classes (\\MlstPairEemmNTestPerSide{} per side). A mark applies to the "
              "configuration.__ZERO__"),
    "mlp": ("tab:mlp", "Nonlinear probes on the frozen features of networks pretrained on JetClass-II with 188, 162, 43 and 17 classes and of the controls, trained and scored "
            "on each task (a multilayer perceptron with one hidden layer of \\LpMlpHidden{} units, "
            "\\LpMlpSeedsWord{} seeds averaged) at the largest training set: class-balanced JetClass-II AUC (all, resonant and QCD "
            "classes) with the ratio $\\rho$ of $1-\\mathrm{AUC}$, equation~(\\ref{eq:ratio}), to the \\Kfull-class run of the same seed, JetClass AUC, and AUC and $R_{30}$ "
            "on top, quark/gluon and Herwig jets (missing inputs set to zero); then the class pairs of "
            "table~\\ref{tab:pairs}, class-balanced, except $X\\to ee$ vs $\\mu\\mu$, which has no nonlinear probe. "
            "The models with the mass regression and the numbered random partitions have no nonlinear probe on the "
            "class-balanced test set: they show a dash in its columns and are left out of the pair blocks; the next block gives "
            "them on the natural-composition test set beside the levels of the hierarchy on that set, and no $\\rho$ "
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
           "pretrained networks use their checkpoints as trained, so this check uses the natural-composition test set. " + SPREAD),
    "balance": ("tab:balance", "Linear probes on the frozen features of networks pretrained on JetClass-II with 188, 162, 43 and 17 classes and of the controls, trained on "
                "$10^{6}$ JetClass-II jets and scored on the natural-composition test "
                "set and on the class-balanced one: AUC averaged over the classes and the ratio $\\rho$ of $1-\\mathrm{AUC}$, equation~(\\ref{eq:ratio}), to "
                "the \\Kfull-class run of the same seed. " + SPREAD),
    "jc1_class_auc": ("tab:jc1_class_auc", "Networks pretrained on JetClass-II with 188, 162, 43 and 17 classes and the "
                      "controls, trained on one million JetClass jets: the area under the curve of each JetClass class "
                      "against the other nine. Upper part: fine-tuned, beside the network trained from scratch. Lower "
                      "part: linear probes on the frozen features, beside the random network without pretraining whose "
                      "BatchNorm statistics are recomputed. " + SPREAD + " From scratch: three training seeds."),
    "fill_main": ("tab:fill_main", "Linear probes on the frozen features of networks pretrained on JetClass-II with 188, 162, 43 and 17 classes, with the inputs a "
                  "data set lacks set to zero, to the training mean or to random draws of a real particle, trained at the "
                  "largest training set. Upper part: relative "
                  "drop in JetClass-II accuracy (percent) when the slots the top or the quark/gluon data lack are "
                  "substituted, against the same model with every input. Lower part: $R_{30}$ on top and quark/gluon "
                  "tagging. Mean and SD over three runs. The \\Kfull-class model with kinematic inputs only has nothing missing (one run). The controls "
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
             "centered on the training median of each of the \\Kfull{} classes, a truth-label input the soft-drop-mass "
             "row does not have. A multilayer perceptron is used because the mass is not linear in the features "
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
    "sculpt": ("tab:sculpt", "Mass sculpting by the output scores of networks pretrained on JetClass-II with 188, 162, 43 and 17 classes, with 162 and 17 classes plus a mass regression__SCULPT_FLAV__, on the JetClass-II test jets: Jensen--Shannon distance (base 2) $\\times10^{2}$ between the "
               "$m_{\\mathrm{SD}}$ distributions (__SCULPT_BINS__) of all QCD test jets and of those passing a cut on "
               "$1-g_{\\mathrm{QCD}}$ at the QCD efficiency $\\epsB$ given; lower is less sculpting.__SCULPT_SIG__ The last row is the "
               "distance expected for a random selection of as many of the QCD test jets (drawn without replacement). "
               + SPREAD_MASS),
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
    "per_class": ("tab:perclass", "The AUC of each JetClass-II class against the rest, as a linear probe on the frozen "
                  "features and fine-tuned, for every network of the per-class figures."),
    "per_run": ("tab:perrun", "Each run of networks pretrained on JetClass-II with 188, 162, 43 and 17 classes and of the controls, and of the network trained from scratch, at "
                "the largest training set of each task: linear probe on the frozen features and fine-tuning. Linear probes on top and quark/gluon with the missing inputs substituted as chosen on "
                "validation (\\LpSubstitutionTop{} for top, \\LpSubstitutionQg{} for quark/gluon); fine-tuning with "
                "them set to zero. JetClass-II: class-balanced test set. The \\Kfull-class model with kinematic inputs only has "
                "one run, on top only; from scratch has no linear probe (dashes)."),
    "rand_prong": ("tab:randprong", "Networks pretrained on JetClass-II with random partitions into 17 classes, fine-tuned on JetClass: __PRONG__ the groups that mix prong counts, and "
                   "each partition's fine-tuned rejection of QCD per signal class over that of the \\Kfull-class run of the "
                   "same seed. $^\\dagger$: fewer than $\\nmin$ QCD jets pass in the partition's run or in that "
                   "\\Kfull-class run (equation~(\\ref{eq:cut}))."),
    "lrsweep": ("tab:lrsweep", "Learning-rate choice for training Sophon's network, pretrained on JetClass-II with "
                "\\Kfull{} classes (seed 1) or trained from scratch, on $10^{4}$ top and quark/gluon jets: the best validation "
                "accuracy of each rate on top and quark/gluon and their sum. Pretrained: the rate of the pretrained layers, with the new "
                "output head at 50 times that rate, at most $5\\times10^{-3}$. $^{\\ast}$: the rate chosen, on the sum for the pretrained "
                "weights and on each task's accuracy for the networks trained from scratch; the test jets play no part."),
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
    "xbx_node": ("tab:xbxnode", "Resonance tagging, the $X\\to bs$ closest-class output: rejection of QCD jets."),
    "published": ("tab:published", "Comparison with published models on JetClass, top tagging and quark/gluon tagging. "
                  "Our rows are the network trained from scratch and fine-tuned from the network pretrained with "
                  "\\Kfull{} classes, as the mean $\\pm$ SD over three runs, with the inputs that the top and "
                  "quark/gluon data lack set to zero. The row with kinematic inputs is a \\Kfull-class network "
                  "pretrained and fine-tuned on kinematic inputs only. Training jets counts the task's own training "
                  "set. Models named f.t., and OmniLearned, were pretrained on other data. AUC is the area under the "
                  "receiver operating characteristic curve, on JetClass averaged over the classes, each against the "
                  "rest, which with JetClass's equal class sizes equals the average over class pairs that ParT "
                  "reports. $R_{50}$ and $R_{30}$ are the inverse background efficiencies at 50\\% and 30\\% signal "
                  "efficiency, with quarks as the signal in quark/gluon tagging. The per-class rejections on JetClass "
                  "are in table~\\ref{tab:jetclassrej}."),
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
    "summary_auc": ("tab:summaryauc", "The AUC beside each headline metric of table~\\ref{tab:summary}."),
    "compute": ("tab:compute", "GPU types and GPU-hours of every pretraining run, summed over the training and "
                "validation passes of its epochs as each epoch recorded its device, and the GPU types its "
                "fine-tuning ran on."),
}
# (caption key, table file, wide, caption fill) of the tables written as placeholders until their data exist
PLACEHOLDERS = (("probes", "mlst_probes", True, None), ("probes_n", "mlst_probes_n", True, None),
                ("probes_n_full", "mlst_probes_n_full", True, None), ("ft_full", "mlst_ft_full", True, None),
                ("pairs", "mlst_pairs", True, {"__ZERO__": ""}), ("mlp", "mlst_mlp", True, None), ("bn", "mlst_bn", False, None),
                ("balance", "mlst_balance", False, None), ("fill", "mlst_fill", True, None), ("fill_main", "mlst_fill_main", True, None),
                ("jc1_class_auc", "mlst_jc1_class_auc", True, None),
                ("mass", "mlst_mass", False, None), ("mass_detail", "mlst_mass_detail", True, {"__BLANK__": ""}),
                ("sculpt", "mlst_sculpt", True, {"__SCULPT_BINS__": r"\pending{bins}", "__SCULPT_FLAV__": "", "__SCULPT_SIG__": "", "__SCULPT_CUT__": "", "__SCULPT_RUNS__": ""}), ("sculpt_boot", "mlst_sculpt_boot", False, {"__REPLICAS__": r"\pending{n}"}), ("ft", "mlst_ft", True, None),
                ("ft_rej", "mlst_ft_rej", True, {"__FEWEST__": r"\pending{n}"}), ("boot", "mlst_boot", True, {"__JC2COMP__": "class-balanced"}),
                ("bench_top", "mlst_bench_top", True, None), ("bench_qg", "mlst_bench_qg", True, None),
                ("xbx", "mlst_xbx", False, None), ("xbx_rej", "mlst_xbx_rej", True, None), ("xbx_node", "mlst_xbx_node", False, None),
                ("rand_prong", "mlst_rand_prong", True, {"__PRONG__": ""}),
                ("lrsweep", "mlst_lrsweep", False, None), ("inputs", "mlst_inputs", True, None),
                ("partrepro", "mlst_partrepro", False, None), ("compute", "compute", False, None),
                ("jetclass_all", "mlst_jetclass_all", True, None), ("jetclass_probe", "mlst_jetclass_probe", True, None),
                ("jetclass_nbkg", "mlst_jetclass_nbkg", True, None),
                ("coarse", "mlst_coarse", True, None), ("qcdhf", "mlst_qcdhf", False, {"__QCDHF_C__": r"\pending{n}", "__QCDHF_LIGHT__": r"\pending{n}"}),
                ("summary", "mlst_summary", False, None), ("summary_auc", "mlst_summary_auc", True, None),
                ("masseffect", "mlst_masseffect", True, None), ("bench_qg_herwig", "mlst_bench_qg_herwig", False, None),
                ("published", "mlst_published", True, None), ("labelsets", "mlst_labelsets", True, None),
                ("probes_full", "mlst_probes_full", True, None), ("pairs_full", "mlst_pairs_full", True, {"__ZERO__": ""}),
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
    ("MlstFtJcOneErrCoarseOverFullBootEFour", "the JetClass bootstrap at 1e4 (after the equal-stopping read-out)", ""),
    ("DesignNTrainSelected", "a committed count of the selected pretraining jets", ""),
    ("FtBudgetEvalSmall", "build_ft_budget_jobs.py", ""),
    ("LpSubstitutionTop", "fill fits", ""), ("LpSubstitutionQg", "fill fits", ""),
    ("MlstFtJcOneCoarseNRejDrop", "JetClass curves", ""), ("MlstFtJcOneCoarseRejDropClasses", "JetClass curves", ""),
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
    ("MlstFtQgRthirtyCoarseOverFull", "fine-tuning read-outs", "Pr"),
    ("MlstFtQgRthirtyHerwigOverPythiaFull", "fine-tuning read-outs", "Pr"),
    ("MlstFtQgRthirtyHerwigOverPythiaScratch", "fine-tuning read-outs", "Pr"),
    ("MlstPartreproRthirtyRescaleOverPart", "partrepro", "Pr"), ("MlstPartreproSliceDropRatio", "not run", ""),
    ("MlstLpJcTwoTopZeroAccDropFull", "fits_v4", "Pr"), ("MlstLpJcTwoQgZeroAccDropFull", "fits_v4", "Pr"),
    ("MlstLpTopRthirtyDrawOverZeroFull", "fill fits", "Pr"), ("MlstLpTopRthirtyKinOverFull", "kinematic-only probe", "Pr"),
    ("MlstLpTopRthirtyFullOverInitBn", "fill fits", "Pr"), ("MlstLpQgRthirtyFullOverInitBn", "fill fits", "Pr"),
    ("MlstLpTopRthirtyFullOverInitBnZero", "probes", "Pr"), ("MlstLpQgRthirtyFullOverInitBnZero", "probes", "Pr"),
    ("MlstLpJcOneGapFracEFour", "fixed-budget fine-tuning (budget_metrics.json)", "Pr"),
    ("MlstMassHeadResMassOverUncorr", "head/atlas_res", "Pr PrMax"),
    ("MlstMassHeadResMassOverMlpRes", "head/atlas_res", "Pr PrMax"),
    ("MlstCmpMlpResMassOverRes", "head/atlas_res", "Pr PrMax"), ("MlstCmpMlpCoarseOverFull", "head/atlas_res", "Pr"),
    ("MlstMassHeadResMassResponse", "head/atlas_res", ""), ("MlstMassHeadCoarseMassResponse", "head/atlas_res", ""),
    ("MlstMassUncorrectedResponse", "head/atlas_res", ""),
    ("MlstMassRespAxesShareQcd", "head/atlas_res", ""),
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
BN_DEF = (" The random network with BatchNorm recomputed has the running mean and variance of its BatchNorm layers "
          "re-estimated on \\DesignBnJets{} training jets")
BN_DEF_BOTH = BN_DEF + ", which the other random network leaves at their untrained defaults."
# the captions of the paper's tables (PI 2026-11-12: what the table compares, then the definitions and run counts
# needed to read it, in sentences; every mark of the table explained, and no other)
PNETS = "the networks pretrained with 188, 162, 43 and 17 classes"
PART_DEF = " A flavor boundary is a class boundary between decays that differ only in quark flavor."
R_DEF = " $R_{50}$ and $R_{30}$ are the inverse background efficiencies at 50\\% and 30\\% signal efficiency"
RUNS_OF = {"set": "\\MlstNRunsWord{} pretraining runs per label set", "flav": "\\RandFlavRuns{} per flavor-boundary partition",
           "mass": "\\MlstMassNRunsWord{} per network with an added mass output",
           "rand": "one for each of five random partitions", "init": "three initializations of the random network",
           "scratch": "three training seeds from scratch"}


def _runs(*kinds) -> str:
    """The run-count sentence of a caption, one item per kind of row (a key of RUNS_OF, or the item itself)."""
    items = [RUNS_OF.get(k, k) for k in kinds]
    return " Values are the mean $\\pm$ standard deviation over " + ", ".join(items[:-1]) + " and " + items[-1] + "."


# a mark that only some tables carry: its sentence is added when a cell of the printed table has it
KEY_MARKS = {
    "probes": (("^{*}", " $^{*}$ marks a probe that did not converge."),),
    "qcdhf": ((r"\dagger", " $^\\dagger$ marks a value for which fewer than \\MlstMinBkgPass{} jets pass in a run."),),
    "jetclass_rej": ((r"\dagger", " $^\\dagger$ marks a value for which fewer than \\MlstMinBkgPass{} QCD jets pass in a "
                                  "run."),),
    "lrsweep": ((r"^{\ast}", " $^{\\ast}$ marks the rate chosen, by the sum for fine-tuning and by each task's accuracy "
                             "from scratch."),),
    "pairs": ((r"\dagger", " In the linear-probe rows, $^{\\dagger}$ marks a pair that the label set merges into one "
                           "class."),
              (r"\ddagger", " $^{\\ddagger}$ marks a pair whose first side it merges with part of the second."),
              (r"\parallel", " $^{\\parallel}$ marks a pair that some of the random partitions merge.")),
}
# one definition and one wording of a multi-class AUC in every caption (PI 2026-11-12); mlst_v4.jetclass_rows stops the
# build if the JetClass average over class pairs, ParT's, differs from it
AUC_OVR = ("AUC is the area under the receiver operating characteristic curve, averaged over the classes, each against "
           "the rest.")
AUC_JC = " With JetClass's equal class sizes this equals the average over class pairs that ParT reports."
RHO_SHORT = "the ratio $\\rho$ of $1-\\mathrm{AUC}$ to the \\Kfull-class run of the same seed"
CAPTION_SHORT = {
    "probes": "Linear probes on the frozen features of the networks of table~\\ref{tab:summary}, compared on accuracy "
              "and on $R_{50}$, the inverse background efficiency at 50\\% signal efficiency with quarks as the signal "
              "in quark/gluon tagging, in top tagging and in quark/gluon tagging scored on Pythia and on Herwig jets, at "
              "each task's largest training set. Their $R_{30}$ and AUC are in tables~\\ref{tab:summary} "
              "and~\\ref{tab:summaryauc}. The missing top and quark/gluon inputs are replaced by \\LpSubstitutionTop{} "
              "for top and by \\LpSubstitutionQg{} for quark/gluon, as chosen on validation, and set to zero for the "
              "released Sophon checkpoint. The network with kinematic inputs is pretrained on those inputs only."
              + PART_DEF + _runs("set", "flav", "mass", "rand", "init"),
    "probes_n": "Linear probes on the frozen features of the networks of table~\\ref{tab:summary} at the training-set "
                "sizes in the column headers, in the quantities of the training-size figures, accuracy on "
                "class-balanced JetClass-II and on JetClass and $R_{30}$, the inverse background efficiency at 30\\% "
                "signal efficiency, in top tagging and in quark/gluon tagging on Pythia and on Herwig jets. The values "
                "at the largest training sets are in table~\\ref{tab:summary}, and the missing inputs are replaced as "
                "in table~\\ref{tab:probes}." + PART_DEF + _runs("set", "flav", "mass", "rand", "init"),
    "probes_n_full": "Linear probes of table~\\ref{tab:probesn} at every training-set size, with each random "
                     "partition on its own row and the area under the receiver operating characteristic curve (AUC) "
                     "beside each quantity. On JetClass-II and JetClass the AUC is averaged over the classes, each "
                     "against the rest. The network with "
                     "kinematic inputs is pretrained on those inputs only." + _runs("set", "flav", "mass", "rand", "init")
                     + " $^{*}$ marks a probe that did not converge.",
    "per_class": "The area under the receiver operating characteristic curve (AUC) of each JetClass-II class "
                 "against the other classes for " + PNETS + ", with an added mass "
                 "output and with the 17-class partitions, as linear probes on the frozen features trained on one million "
                 "class-balanced JetClass-II jets (upper part) and fine-tuned on as many (lower part). The columns name "
                 "each network by its number of classes, and the classes are grouped by the 17 classes of the coarsest "
                 "label set, decays that differ only in flavor side by side." + PART_DEF
                 + _runs("set", "flav", "mass", "rand"),
    "probes_full": f"Linear probes on the frozen features of {NETS} and of the controls (every random partition), at each task's largest training "
              "set. Upper part: JetClass-II (class-balanced test set) and JetClass. Lower part: top and quark/gluon "
              "tagging (also on Herwig jets). Accuracy, AUC, " + RHO_SHORT + ", and the "
              "background rejections $R_{50}$ and $R_{30}$. On JetClass-II and JetClass the " + AUC_OVR + AUC_JC
              + " On JetClass-II the average is also given over the classes with the full \\JcBalTestPerClass{} "
              "test jets (full count), over the resonant classes and over the QCD classes. "
              "Below the rule: the released Sophon checkpoint, outside the "
              "controlled comparison. " + SD_SHORT + " $^{*}$: a probe that did not converge.",
    "pairs": "Separation of pairs of JetClass-II classes by " + PNETS + ", with an added mass output and with the "
             "17-class partitions, by the two random networks and by training from scratch, as the area "
             "under the receiver operating characteristic curve (AUC). The fine-tuned networks are trained on "
             "$10^{6}$ class-balanced JetClass-II jets and scored on other test jets than the linear probes, by the "
             "summed outputs of the first side's classes over those of both sides. An entry $>0.9999$ is a mean AUC "
             "that rounds to 1.0000." + BN_DEF_BOTH + PART_DEF
             + _runs("set", "flav", "mass", "rand", "init", "scratch"),
    "pairs_full": f"Pairs of JetClass-II classes separated by {NETS} and by the controls (every random partition): the area under the receiver "
             "operating characteristic curve (AUC) of a binary linear probe "
             "on the frozen features and of the network fine-tuned on $10^{6}$ class-balanced JetClass-II jets, "
             "and " + RHO_SHORT + " in the same mode; the two modes are scored on different test jets. An entry "
             "$>0.9999$ is a mean AUC that rounds to 1.0000. "
             "$^{\\dagger}$: the label set merges the two sides into one class; $^{\\ddagger}$: it merges the first "
             "side with part of the second. " + SD_SHORT + "__ZERO__",
    "mlp": "Nonlinear probes (a multilayer perceptron with one hidden layer of \\LpMlpHidden{} units) on the frozen "
           f"features of {NETS} and of the controls, at the largest training set of each task: the quantities of "
           "table~\\ref{tab:probes} and the class pairs of table~\\ref{tab:pairs}; then the models scored on the "
           "natural-composition test set only, and the linear probe on the hidden layer of the pretraining "
           "head beside the one on the class token. " + SD_SHORT,
    "coarse": "Linear probes on the frozen features of " + PNETS + " and of the control networks, compared on "
              "coarser versions of the JetClass-II task. The probes are trained on one million class-balanced "
              "JetClass-II jets whose 188 classes are merged into the 43 or the 17 classes of the pretraining label "
              "sets or into the 17 classes of the partition with no flavor boundary." + PART_DEF + " "
              + AUC_OVR.replace("the classes", "the target's groups") + _runs("set", "flav", "mass", "rand", "init"),
    "coarse_full": f"Linear probes on the frozen features of {NETS} and of the controls (every random partition), trained on one million "
              "class-balanced JetClass-II jets, with the classes merged into coarser targets: the 43 and the 17 "
              "groups of the class hierarchy and the 17 groups of the partition with no flavor boundary. "
              + AUC_OVR.replace("the classes", "the target's groups") + " " + SD_SHORT,
    "qcdhf": "Rejection of QCD jets by the $X\\to b\\bar b$ discriminant of linear probes on the frozen features of "
             + PNETS + " and of the control networks, trained on one million class-balanced JetClass-II jets. The "
             "discriminant is the probe's $X\\to b\\bar b$ score divided by the sum of that score and the QCD scores. "
             "$R_{50}$ is the inverse QCD efficiency at 50\\% $X\\to b\\bar b$ efficiency, for QCD jets with two or "
             "more $b$ quarks, with one $b$ quark, and for all QCD jets." + PART_DEF
             + _runs("set", "flav", "mass", "rand", "init"),
    "qcdhf_full": f"Linear probes on the frozen features of {NETS} and of the controls (every random partition), trained on one million "
             "class-balanced JetClass-II jets: rejection $R_{50}$ of Sophon's $X\\to b\\bar b$ discriminant at "
             "50\\% $X\\to b\\bar b$ efficiency for QCD jets with two or more $b$ quarks, with one $b$ quark, and "
             "for all QCD jets. " + SD_SHORT + " $^\\dagger$: fewer than \\MlstMinBkgPass{} jets pass in a run.",
    "bn": f"Linear probes on the frozen features of {NETS} and of the controls, on JetClass-II at its natural "
          "composition, with the BatchNorm statistics as accumulated in pretraining and recomputed on training "
          "jets: the AUC and " + RHO_SHORT + ". " + AUC_OVR + " " + SD_SHORT,
    "jc1_class_auc": "The area under the receiver operating characteristic curve (AUC) of each JetClass class "
                     "against the other nine, for " + PNETS + ", with an added mass "
                     "output and with the 17-class partitions, fine-tuned on one million JetClass jets beside the same "
                     "architecture trained from scratch, and as linear probes on the frozen features, trained on as "
                     "many jets, beside the random network." + PART_DEF
                     + _runs("set", "flav", "mass", "rand", "init", "scratch"),
    "fill": "Linear probes on the frozen features of " + PNETS + ", with an added mass output and with the 17-class "
            "partitions, and of the random network, compared between "
            "substitutions for the inputs that the top and quark/gluon data lack. $R_{30}$ is the inverse background "
            "efficiency at 30\\% signal efficiency. The missing inputs are set to zero, to the training mean (mean) or "
            "to random draws of a real particle (draws), and the substitution chosen on validation for each task is "
            "in table~\\ref{tab:probes}." + BN_DEF + "." + PART_DEF + _runs("set", "flav", "mass", "rand", "init"),
    "fill_full": f"Linear probes on the frozen features of {NETS} and of the controls (every random partition), with the inputs a data set lacks "
            "set to zero, to the training mean or to random draws of a real particle: the relative drop in "
            "JetClass-II accuracy (percent) when those slots are substituted, then $R_{30}$ on top and "
            "quark/gluon tagging. " + SD_SHORT,
    "mass": "Jet-mass resolution and median response on JetClass-II, compared between the soft-drop mass, the mass "
            "output of the pretraining head and a multilayer perceptron that regresses the mass from the frozen "
            "features. The response $r = m/m_{\\mathrm{gen}}$ is the estimated mass over the generator-level soft-drop "
            "mass, and the resolution is half the range between the 15.9\\% and 84.1\\% quantiles of $r$ divided by its "
            "median, over all test jets with a generator-level match. The perceptron's target is centered on the "
            "training median of the jet's true class, which the other two estimates are not."
            + _runs("set", "mass", "init"),
    "boot": f"Linear probes on the frozen features and fine-tuning of {NETS}: uncertainty (percent) of the comparisons "
            "between their label sets at the largest training set, as the SD of the logarithm of "
            "the ratio over the pretraining runs / that from the finite test sample. JetClass-II: the ratio of "
            "$1-\\mathrm{AUC}$; top and quark/gluon: the ratio of $R_{30}$.",
    "lrsweep": "Best validation accuracy at each learning rate tried for fine-tuning the network pretrained with "
               "\\Kfull{} classes and for training from scratch, on $10^{4}$ top-tagging jets and on $10^{4}$ "
               "quark/gluon jets, with the sum of the two in the last column. For fine-tuning, the rate is that of "
               "the pretrained layers, and the new output head trains at 50 times that rate, at most "
               "$5\\times10^{-3}$.",
    "mass_detail": "Jet-mass resolution and median response on JetClass-II in detail, defined as in "
                   "table~\\ref{tab:mass}, for the soft-drop mass, the mass output of the pretraining head and the "
                   "multilayer-perceptron and ridge regressions on the frozen features of " + PNETS + ", of the networks "
                   "with an added mass output and of the random network. The upper parts give the values over resonant "
                   "and QCD jets, and for the ridge regression also over all jets, whose other values are in "
                   "table~\\ref{tab:mass}. The lower parts give the ten bins of jet $p_{\\mathrm{T}}$ and five of "
                   "generator-level soft-drop mass that the binned mass figures draw, and __BLANK__"
                   + _runs("set", "mass", "init"),
    "sculpt": "Mass sculpting of QCD jets by the output scores, used directly, of " + PNETS + ", with an added mass "
              "output and with the 17-class partitions with no flavor boundary, a $b$-quark boundary or a random "
              "flavor boundary. The "
              "distance is the Jensen--Shannon distance (base 2) $\\times10^{2}$ between the soft-drop mass "
              "distributions of all QCD test jets and of those passing a cut on $1-g_{\\mathrm{QCD}}$, one minus the "
              "summed output of the QCD classes. The cut keeps the fraction $\\epsB$ of the QCD jets__SCULPT_CUT__. "
              "The last row is the distance expected for a random selection of as many QCD test jets."
              "__SCULPT_RUNS__",
    "sculpt_boot": "Differences in mass sculpting between pairs of pretrained networks, each row the first network "
                   "named minus the second. $\\Delta$ is the difference of the Jensen--Shannon distances of "
                   "table~\\ref{tab:sculpt} ($\\times10^{2}$), averaged over pairs of pretraining runs with the same "
                   "seed. Run SD is the standard deviation of the paired differences over the seeds, bootstrap SD is "
                   "that of $\\Delta$ over __REPLICAS__ bootstrap replicas of the test jets, and $\\sigma$ adds the two "
                   "in quadrature. The blocks are the cuts of table~\\ref{tab:sculpt}, which keep 1\\% and 10\\% of the "
                   "QCD jets ($\\epsB$) or half of the resonant jets ($\\epsS = 50\\%$)." + PART_DEF + " There are "
                   "\\MlstNRunsWord{} pairs of runs for a label set against 188 classes and two for the others. "
                   "$^{\\dagger}$ marks a pair that the bootstrap does not include, whose $\\sigma$ is the run SD "
                   "alone.",
    "ft": "Fine-tuning of the networks of table~\\ref{tab:summary}, and training from scratch, at the training-set sizes "
          "in the column headers, in the quantities of the training-size figures, accuracy on class-balanced "
          "JetClass-II and on JetClass and $R_{30}$, the inverse background efficiency at 30\\% signal efficiency, in "
          "top tagging and in quark/gluon tagging on Pythia and on Herwig jets. The values at the full training sets "
          "are in table~\\ref{tab:summary}, the 17-class partitions are fine-tuned on JetClass-II at one million jets "
          "only, and the missing top and quark/gluon inputs are set to zero." + PART_DEF
          + _runs("set", "flav", "mass", "rand", "scratch"),
    "ft_full": "Fine-tuning of table~\\ref{tab:ft} at every training-set size, with each random partition on its own "
               "row and the area under the receiver operating characteristic curve (AUC) beside each quantity. The "
               "JetClass-II blocks add the AUC averaged over the QCD classes and $\\rho$, $1-\\mathrm{AUC}$ at $10^{6}$ "
               "jets over that of the \\Kfull-class run of the same seed. On JetClass-II and JetClass the AUC is "
               "averaged over the classes, each against the rest. A dash marks a size at "
               "which the network was not fine-tuned. The network with kinematic inputs is pretrained and fine-tuned "
               "on those inputs only." + _runs("set", "flav", "mass", "rand", "scratch"),
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
                "table~\\ref{tab:benchqgherwig}.",
    "jetclass": "JetClass: Sophon's network pretrained on JetClass-II with 188, 162, 43 and 17 classes and fine-tuned, or "
                "trained from scratch, on one million JetClass jets, beside published models (2 million jets, from "
                "scratch): the number of trainable parameters, accuracy and AUC. " + AUC_OVR + AUC_JC + " Ours: mean "
                "$\\pm$ SD over three pretraining runs; from scratch: three seeds. Per-class rejections in "
                "table~\\ref{tab:jetclassrej}.",
    "jetclass_rej": "Rejection of QCD jets for each JetClass signal class by " + PNETS + " and with an added mass "
                    "output, fine-tuned on one million JetClass jets, by the same architecture trained from scratch on those jets, and by "
                    "published models trained from scratch on two million jets. The rejection is the inverse QCD "
                    "efficiency at the signal efficiency in the subscript, 50\\% except __WPHQQL__\\% for "
                    "$H\\to\\ell\\nu qq'$ and __WPTBL__\\% for $t\\to b\\ell\\nu$. Values of this work are the mean "
                    "$\\pm$ standard deviation over \\MlstNRunsWord{} pretraining runs per label set, "
                    "\\MlstMassNRunsWord{} per network with an added mass output and three training seeds from "
                    "scratch. Accuracy and AUC are in tables~\\ref{tab:summary}, \\ref{tab:summaryauc} "
                    "and~\\ref{tab:published}.",
    "jetclass_all": f"JetClass fine-tuning as tables~\\ref{{tab:published}} and~\\ref{{tab:jetclassrej}} of {NETS} and of the controls; the row of "
                    "all five random partitions gives the mean $\\pm$ SD over them. " + AUC_OVR + AUC_JC + " " + SD_SHORT
                    + " $^\\dagger$ as in table~\\ref{tab:jetclassrej}.",
    "jetclass_probe": f"Linear probes on the frozen features of {NETS} and of the controls, trained and scored on "
                      "JetClass at the largest training set, in the columns of table~\\ref{tab:jetclassall}. Below "
                      "the rule, the released Sophon checkpoint, outside the controlled comparison. " + AUC_OVR
                      + AUC_JC + " " + SD_SHORT
                      + " $^\\dagger$ as there.",
    "xbx_rej": "Rejection of QCD jets when tagging $X\\to b\\bar b$ and $X\\to bs$ jets in the soft-drop mass window "
               "\\XbxWindowLo--\\XbxWindowHi~GeV, for the networks and read-outs of table~\\ref{tab:xbx}. $R_{50}$ "
               "and $R_{30}$ are the inverse QCD efficiencies at 50\\% and 30\\% signal efficiency. A dash marks a "
               "read-out that the network does not have." + PART_DEF
               + _runs("set", "flav", "mass", "rand", "init", "scratch"),
    "xbx_node": "Rejection of QCD jets when tagging $X\\to bs$ jets with the pretrained output of the class that "
                "contains $X\\to bq$ against the QCD outputs, nothing trained, in the soft-drop mass window of "
                "table~\\ref{tab:xbx}. $R_{50}$ and $R_{30}$ are the inverse QCD efficiencies at 50\\% and 30\\% "
                "signal efficiency." + PART_DEF + _runs("set", "flav", "mass", "rand"),
    "summary": "Linear probes on the frozen features and fine-tuning of " + PNETS + ", with an added mass output "
               "and with the 17-class partitions, compared on accuracy on class-balanced JetClass-II and JetClass and "
               "on $R_{30}$, the inverse background efficiency at 30\\% signal efficiency, in top tagging and in "
               "quark/gluon tagging scored on Pythia and on Herwig jets. Linear probes use each task's largest "
               "training set, and fine-tuning one million JetClass-II and JetClass jets and the full top and "
               "quark/gluon training sets, all of Pythia jets. The released Sophon checkpoint is shown on JetClass-II "
               "and JetClass only, and the AUC of each value is in table~\\ref{tab:summaryauc}." + PART_DEF
               + _runs("set", "flav", "mass", "rand", "init", "scratch"),
    "masseffect": "The networks pretrained with 162 and 17 classes, each without and with an added mass output, "
                  "compared on every task. Each row names a task, a metric and the way the network is used, which is "
                  "the mass output of the pretraining head, a probe of the jet mass or of the task on the frozen "
                  "features, fine-tuning, a classifier trained on the frozen features, the direct $X\\to b\\bar b$ "
                  "discriminant or the network output used directly. Mass resolution and median response are defined "
                  "as in table~\\ref{tab:mass}, and the Jensen--Shannon distance as in table~\\ref{tab:sculpt} but "
                  "not multiplied by 100. Training sets and substitutions of the missing inputs are those of "
                  "table~\\ref{tab:summary}, except that the multilayer-perceptron probe on top tagging has the missing "
                  "inputs at zero. A dash marks a quantity the network does not have." + _runs("set", "mass"),
    "summary_auc": "The area under the receiver operating characteristic curve (AUC) of every linear probe and "
                   "fine-tuned network of table~\\ref{tab:summary}, on the same test and training sets. On JetClass-II "
                   "and JetClass the AUC is averaged over the classes, each against the rest. The released Sophon "
                   "checkpoint is shown on JetClass-II and JetClass only." + PART_DEF
                   + _runs("set", "flav", "mass", "rand", "init", "scratch"),
    "published": "Comparison with published models on JetClass, top tagging and quark/gluon tagging. The rows of this "
                 "work are the mean $\\pm$ standard deviation over three runs, trained from scratch or fine-tuned from "
                 "the network pretrained with \\Kfull{} classes, with the inputs that the top and quark/gluon data lack "
                 "set to zero. The row with kinematic inputs is pretrained and fine-tuned on kinematic inputs only. "
                 "The models whose names end in f.t., for fine-tuned, and OmniLearned-l are pretrained on other data. AUC is the area "
                 "under the receiver operating characteristic curve, on JetClass averaged over the classes, each "
                 "against the rest, which with JetClass's equal class sizes equals the average over class pairs that "
                 "ParT reports." + R_DEF + ", with quarks as the signal in quark/gluon tagging.",
    "xbx": "Signal efficiency in percent for $X\\to b\\bar b$ and $X\\to bs$ jets against QCD jets in the soft-drop mass "
           "window \\XbxWindowLo--\\XbxWindowHi~GeV, at the QCD efficiency $\\varepsilon_B$ in the header. The "
           "pretrained networks are not fine-tuned. A classifier or a linear probe is trained on their frozen "
           "features, and direct is the pretrained $X\\to b\\bar b$ output against the QCD outputs with nothing "
           "trained. From scratch is the same architecture trained on these samples, and a dash marks a read-out that "
           "a network does not have." + PART_DEF + _runs("set", "flav", "mass", "rand", "init", "scratch"),
    "inputs": "The per-particle inputs of the networks, with a bullet ($\\bullet$) where the top or the quark/gluon "
              "data provide an input and a dash where they lack it. The last column gives, for each input that a data "
              "set lacks, where a zero lies after the input normalization of a network pretrained with \\Kfull{} "
              "classes, in standard deviations of the pretraining particles. The mean substitution sets a missing "
              "input to the running mean of the input normalization. The random-draw substitution gives a particle "
              "of a top jet a type drawn from JetClass-II's frequencies and a charge drawn given the type, and gives "
              "every particle the displacement inputs of a JetClass-II particle of its type, averaged over four "
              "draws.",
    "partrepro": "ParT's top-tagging training script run from scratch, compared unchanged and with one of two "
                 "changes taken from Sophon, which are not combined. The changes are Sophon's output head and "
                 "Sophon's rescaling of every jet's four-momenta to $p_{\\mathrm{T}}=500$~GeV. AUC is the area under "
                 "the receiver operating characteristic curve." + R_DEF + ", and the last column counts the "
                 "background jets that pass the selection of $R_{30}$. ParT's published result and the network of "
                 "this work trained from scratch, which has both changes and the missing inputs set to zero, are in "
                 "table~\\ref{tab:published}.",
    "labelsets": "Fine-tuning of the networks of table~\\ref{tab:summary}, and training from scratch, on the full top and "
                 "quark/gluon training sets of Pythia jets, compared on accuracy and on $R_{50}$, the inverse background "
                 "efficiency at 50\\% signal efficiency with quarks as the signal in quark/gluon tagging, in top tagging "
                 "and in quark/gluon tagging scored on Pythia and on Herwig jets. Their $R_{30}$ and AUC are in "
                 "tables~\\ref{tab:summary} and~\\ref{tab:summaryauc}." + PART_DEF
                 + _runs("set", "flav", "mass", "rand", "scratch"),
}


def printed_caption(key: str) -> str:
    return CAPTION_SHORT.get(key, CAPTIONS[key][1])


TEXT_CM = 15.5      # widest tabular kept whole (estimated): the 15 cm text block, shrunk by a few per cent at most
LABEL_CM = 6.3      # a wider row label wraps, at its commas, in a table that has to be split
SHRINK_OK = 1.0     # nothing is shrunk (PI 2026-11-11); was 1.08 with adjustbox
BODY_SCALE = 12 / 10.95     # the tables are set at the body size, 12 pt (PI 2026-11-12); the width model was calibrated at \\small


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
    generated tables, measured with LaTeX, and scaled by BODY_SCALE)."""
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
    return (0.177 * len(t) + 0.1 * sum(c.isupper() for c in t)) * bold * BODY_SCALE + 0.42      # capitals are wider


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
         "lrsweep": "ft", "probes_n_full": "lp", "ft_full": "ft", "mlp": "mlp",
         "jc1_class_auc": ["ft", "lp"],
         "qcdhf": "lp", "coarse": "lp", "probes_full": "lp", "fill_full": "lp", "qcdhf_full": "lp", "coarse_full": "lp",
         "jetclass": "ft", "jetclass_rej": "ft", "bench_top": "ft", "bench_qg": "ft", "bench_qg_herwig": "ft",
         }


def _headings(tab: str, mode: str | None = None, star: bool = False) -> str:
    """The headings over the rows of a tabular (PI 2026-11-12). mode (a key of HEADING): the italic heading of the
    downstream mode before each run of rows that name a network (a LABEL or from scratch), unless the builder wrote
    one. The partition rows (flavor boundaries, random partitions) indented under the subheading PARTITIONS. A
    rule between two rows does not end a run (the released checkpoint below its rule); a row of anything else does
    (the soft-drop mass, a published model). star: the heading stays on the page of the row after it (a longtable)."""
    known = sorted(set(LABEL.values()) | {SCRATCH_LABEL}, key=len, reverse=True)
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
AGREE = r"$^{\P}$"        # the runs agree to the digits shown: the value once, no "+- 0" and no "<"
MARK_NOTES = ((SINGLE, r" $^{\S}$ marks a value from a single run."),
              (AGREE, r" $^{\P}$ marks a value whose standard deviation over the runs rounds to zero at the digits shown."))


def _mark_notes(cap: str, body: str) -> str:
    """The caption with a sentence for each of SINGLE and AGREE that a cell of the printed body carries."""
    return cap + "".join(note for mark, note in MARK_NOTES if mark in body)


MAX_DECIMALS = 5        # the finest place a column is printed at (0.98560 +- 0.00004)


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
    sds = [v[1] for k, v in items if k == "p" and not _zero_sd(*v)]
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


def _render(kind: str, v: list, place: int) -> str:
    """A token at its column's place: p mean +- SD, or the mean with AGREE where the SD is zero or prints as zero;
    s a single run's value with SINGLE; o a number that is not a run's."""
    bold, kind = kind.isupper(), kind.lower()       # upper case: a cell _bold chose; the mark stays light
    if kind == "p" and not (_zero_sd(*v) or not round(v[1], place)):
        t, mark = _at(v[0], place) + PM + _at(v[1], place), ""
    elif kind == "p":
        t, mark = _at(v[0], place), AGREE
    else:
        t, mark = ", ".join(_at(x, place) for x in v), SINGLE if kind == "s" else ""
    return (_b(t) if bold else t) + mark


def uniform_columns(tab: str) -> str:
    """The value tokens of a tabular (one cell per line, as the builders write them) printed at one precision
    per column (_column_place), the standard deviation included."""
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
    place = {c: _column_place(v) for c, v in items.items()}
    cells = {}
    for i, j, col in where:
        parts = cells.setdefault(i, _depth0_split(lines[i], "&"))
        parts[j] = TOKEN.sub(lambda m: _render(m.group(1), [float(x) for x in m.group(2).split()], place[col]), parts[j])
    for i, parts in cells.items():
        lines[i] = "&".join(parts)
    return "\n".join(lines)


# Bold (PI 2026-11-12): in the tables that compare networks, per block and column the best value and every value not
# resolved from it at three standard deviations. lower / unity: the columns (0 the row labels) where the smallest
# value, or the one closest to 1, is best (else the largest); skip: columns never bold (counts); published: the
# table has published rows, which take part by the value and the standard deviation they print (none: zero); whole:
# the tabular is one block whatever its headings; blocks: the caption says "each block of a column".
BOLD = {"summary": {"blocks": True}, "summary_auc": {"blocks": True}, "published": {"blocks": True, "published": True, "skip": (1, 2)},
        "labelsets": {}, "probes": {}, "pairs": {"blocks": True}, "probes_n": {"blocks": True}, "ft": {"blocks": True},
        "jc1_class_auc": {"blocks": True},
        "mass": {"blocks": True, "lower": (1,), "unity": (2,)}, "sculpt": {"lower": (1, 3, 6)},
        "jetclass_rej": {"whole": True, "published": True, "skip": (1,)},
        "xbx": {}, "xbx_rej": {}, "coarse": {}, "qcdhf": {}, "fill": {},
        "masseffect": {"of": "a row"}}       # built transposed: each of its rows is a column to _bold
BOLD_NOTE = " Bold marks the best value of {} and every value within three standard deviations of it."
NEVER_BOLD = (LABEL[INIT], LABEL[INIT_BN], LABEL[SOPHON])       # reference rows; so is a number that is no run's (token o)
BOLD_SIGMA = 3.0
_NUM = r"\d+(?:\{,\}\d{3})*(?:\.\d+)?"
PLAIN = re.compile(rf"({_NUM})(?:{re.escape(PM)}({_NUM}))?")


def _bold(key: str, tab: str, spec: dict | None = None) -> str:
    """The rows of a table (a tabular, or the rows of one block of it) with the cells to print bold marked, from
    the unrounded values of their tokens: an upper-case token kind, which _render prints bold (a published cell is
    wrapped here). A block is the rows under one heading across the table (PARTITIONS is no such heading). Per
    block and column, the best value and every value with |mean - best| <= BOLD_SIGMA * sqrt(SD^2 + SD_best^2); a
    single run has SD 0. No bold in a block's column with fewer than two values, or with a value printed as a bound
    (NEAR_ONE), whose best is not a printed value. spec: the settings in place of BOLD[key] (tab:masseffect, whose
    directions follow its rows)."""
    spec = spec or BOLD[key]
    lines = tab.split("\n")
    found, void = collections.defaultdict(list), set()      # (block, column) -> [(line, cell, mean, SD)]
    block, body = 0, r"\toprule" not in tab
    for i, ln in enumerate(lines):
        t = ln.strip()
        if not body:
            body = t.startswith(r"\midrule")
            continue
        parts = _depth0_split(ln, "&")
        if len(parts) == 1:
            block += t.startswith(r"\multicolumn") and PARTITIONS not in t and not spec.get("whole")
            continue
        if parts[0].strip().removeprefix(r"\quad ").strip() in NEVER_BOLD:
            continue
        col = 0
        for j, c in enumerate(parts):
            toks = TOKEN.findall(c)
            if col in spec.get("skip", ()):
                pass
            elif NEAR_ONE in c:
                void.add((block, col))
            elif len(toks) == 1 and toks[0][0] in "ps":
                v = [float(x) for x in toks[0][1].split()]
                found[block, col].append((i, j, v[0], v[1] if toks[0][0] == "p" else 0.0))
            elif spec.get("published") and (m := PLAIN.fullmatch(c.strip().removesuffix(r"\\").strip())):
                mean, sd = (float(x.replace("{,}", "")) if x else 0.0 for x in m.groups())
                found[block, col].append((i, j, mean, sd))
            mc = re.match(r"\s*\\multicolumn\{(\d+)\}", c)
            col += int(mc.group(1)) if mc else 1
    chosen = collections.defaultdict(set)
    for (block, col), cells in found.items():
        if (block, col) in void or len(cells) < 2:
            continue
        score = ((lambda x: abs(x - 1)) if col in spec.get("unity", ()) else (lambda x: x) if col in spec.get("lower", ())
                 else (lambda x: -x))
        *_, best, sd0 = min(cells, key=lambda c: score(c[2]))
        for i, j, mean, sd in cells:
            if abs(mean - best) <= BOLD_SIGMA * math.hypot(sd, sd0):
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


PORTRAIT_CM = 14.6      # the text block (15 cm) less the estimate's error: a tabular estimated wider is set in landscape, never shrunk or cut to fit
LANDSCAPE_CM = 23.6     # the text block turned (24.7 cm) less the estimate's error: only a tabular wider than this is cut into blocks, each that wide
NO_WRAP = 99.0          # label_cm of a row-label column that never wraps


def _stretch(tab: str) -> str:
    """A tabular set across the whole line (tabular* with the space dealt between its columns), so that every table
    and every block of a table has one width and one left edge (PI 2026-11-11)."""
    return re.sub(r"\\begin\{tabular\}\{((?:[^{}]|\{[^{}]*\})*)\}",
                  lambda m: r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}" + m.group(1) + "@{}}",
                  tab, count=1).replace(r"\end{tabular}", r"\end{tabular*}")


def _turned_long(blocks: list, cap: str, label: str, tail: list, whole: bool = False, sep: list = ()) -> str:
    """Blocks of a table too wide for the upright page, in landscape across the whole line. A block that fits a turned
    page stays whole on one (no stub of a few rows under a repeated heading, no heading stranded at the foot of a
    page); a block taller than the page is a longtable whose rows break under its repeated heading. The caption
    above the first block, "Table N (continued)" above every later block and page; one table number. whole: every
    block is a page its builder chose and stays whole whatever the estimate."""
    cont = rf"\makebox[\linewidth][l]{{\textit{{Table~\ref{{{label}}} (continued)}}}}"
    out = [r"\begin{landscape}", "{", LT_SETUP, *sep]
    for k, b in enumerate(blocks):
        fits = whole or _height_cm([b], "") + (0.45 * (len(cap) / 150 + 1) if k == 0 else 0.6) <= LAND_FULL_CM
        if fits:
            if whole and k:         # a page its builder chose never shares one with the block before it
                out.append(r"\newpage")
            # no skip above a caption that heads the page (a longtable's has none either): ten points more for the
            # rows of a page its builder chose (tab:labelsets' first page was 6 pt over with the bold sentence)
            head = ([r"\setlength{\abovecaptionskip}{0pt}", rf"\captionof{{table}}{{{cap}}}", rf"\label{{{label}}}", r"\par\medskip"] if k == 0
                    else [cont, r"\par\smallskip"])
            out += [r"\begin{center}", r"\begin{minipage}{\linewidth}\centering",
                    *head, _stretch(b), r"\end{minipage}", r"\end{center}"]
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
        heads = [i for i, ln in enumerate(rows.split("\n")) if re.match(r"\\multicolumn\{\d+\}\{(?:@\{\})?l\}\{\\textit\{", ln.strip())]
        if len(heads) > 1:
            lines = rows.split("\n")
            for i in reversed(heads[1:]):
                lines.insert(i, r"\newpage")
            rows = "\n".join(lines)
        more = rf"\multicolumn{{{len(spec)}}}{{@{{}}l}}{{\textit{{Table~\ref{{{label}}} (continued)}}}} \\[\smallskipamount]"
        # a later block on a fresh page, so that its first heading is never left behind on the page before
        # (the first block's caption as far above its table as that of a block set whole: a medskip, added once)
        out += ([r"\newpage", r"\addtocounter{table}{-1}"] if k else [r"\addtolength{\belowcaptionskip}{\medskipamount}"]) + [      # a longtable steps the counter: one number a table
                r"\begin{longtable}{@{\extracolsep{\fill}}" + spec + "@{}}",
                (rf"\caption{{{cap}}}\label{{{label}}}\\" if k == 0 else more), r"\toprule", head, r"\midrule", r"\endfirsthead",
                more, r"\toprule", head, r"\midrule", r"\endhead", r"\bottomrule", r"\endlastfoot", rows, r"\end{longtable}"]
    return "\n".join(out + [*tail, "}", r"\end{landscape}", ""])


def _float(key: str, body: str, wide: bool = False, fill: dict | None = None, notes: list | None = None,
           label_cm: float = NO_WRAP, budget: float = PORTRAIT_CM, portrait_label_cm: float | None = None,
           colsep_pt: float = 4.0, pages: bool = False, whole: str | None = None, size: str = "") -> str:
    """A generated table (PI 2026-11-11): at the full text width; a tabular wider than the text block is set in
    landscape rather than cut into narrow blocks or shrunk; a row label never wraps; nothing is scaled. Only a
    tabular wider than the landscape block is cut (split_wide), into blocks of that one width. label_cm and budget
    are kept for the callers that passed them and no longer narrow anything. portrait_label_cm: upright and whole
    whatever the estimate, the row labels wrapping at that width, 0 never (PI 2026-11-12: top, quark/gluon; one
    block). pages: the body's %SPLIT% parts are the pages of a table that no single page holds (tab:summary,
    tab:published, tab:labelsets), each set whole under the caption or "Table N (continued)". whole ("upright" or "turned"): that
    orientation with no part cut, whatever the width estimate says, with the column padding at colsep_pt: for the
    paper's tables whose width was measured with LaTeX at the body size (the text block is 15.0 cm upright and
    24.7 cm turned). size: a type size below the body's for a table that no orientation holds at the body size
    without breaking another rule of the brief (tab:probes; PI 2026-11-12: nothing else is below the
    body size); the caption stays at the body size."""
    env = "table*" if wide else "table"
    label, cap = CAPTIONS[key][0], printed_caption(key)
    for k, v in (fill or {}).items():
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
    parts = [_labels(uniform_columns(_bold(key, _headings(b.strip(), _mode(key, k)))) if key in BOLD and V4.TOK_OPEN in b
                     else _headings(uniform_columns(b.strip()), _mode(key, k))) for k, b in enumerate(body.split("%SPLIT%"))]
    parts = [re.sub(r"\\mbox\{((?:[^{}]|\{[^{}]*\})*)\}", r"\1", t) if "begin{tabular}" in t else t for t in parts]
    if key in BOLD and any(BOLD_ON in t for t in parts):
        cap += BOLD_NOTE.format(BOLD[key].get("of") or ("each block of a column" if BOLD[key].get("blocks") else "a column"))
    cap += "".join(note for mark, note in KEY_MARKS.get(key, ()) if mark in body)
    cap = _mark_notes(cap, "\n".join(parts))
    tail = [rf"\par\smallskip{{{n}}}" for n in notes or []]
    if portrait_label_cm is not None:
        lab = (r">{\raggedright\arraybackslash}p{" + f"{portrait_label_cm:.1f}" + "cm}") if portrait_label_cm else "l"
        boxes = [_stretch(re.sub(r"^(\s*\\begin\{tabular\}\{)l", lambda m: m.group(1) + lab, t, count=1)) for t in parts]
        # the column padding 4 pt (LaTeX's default 6 pt): the type stays at the tables' size, nothing is scaled
        return "\n".join([rf"\begin{{{env}}}[t]", r"\centering", rf"\setlength{{\tabcolsep}}{{{colsep_pt:g}pt}}",
                          rf"\caption{{{cap}}}", rf"\label{{{label}}}", *([size] if size else []),
                          "\n\\par\\bigskip\n".join(boxes), *tail, rf"\end{{{env}}}", ""])
    cut = {o: [t.strip() for b in parts for t in ([b] if whole else split_wide(b, w, NO_WRAP))]
           for o, w in (("upright", PORTRAIT_CM), ("turned", LANDSCAPE_CM))}
    # upright when nothing is cut; turned when that keeps the table whole; a table cut either way takes the
    # orientation with fewer pages (upright on a tie)
    one_turned_page = len(cut["turned"]) == len(parts) and _height_cm(cut["turned"], cap) <= LAND_PAGE_CM
    # turned blocks flow as longtables, their rows breaking across pages; upright blocks stay whole, packed on pages
    flow = math.ceil((sum(_height_cm([b], "") - 0.45 for b in cut["turned"]) + 0.45 * (len(cap) / 150 + 1)) / LAND_FULL_CM)
    upright = whole != "turned" and (len(cut["upright"]) == len(parts) or (
        len(cut["turned"]) > len(parts) and _pages(cut["upright"], cap, PAGE_FULL_CM, 90) <= flow))
    sep = [rf"\setlength{{\tabcolsep}}{{{colsep_pt:g}pt}}"] if whole else []
    if not upright and not one_turned_page:     # wide, and more than a turned page: longtables in landscape
        return _turned_long(cut["turned"], cap, label, tail, whole=pages, sep=sep)
    boxes = [_stretch(t) for t in cut["upright" if upright else "turned"]]
    turn = ([r"\begin{landscape}"], [r"\end{landscape}"]) if not upright else ([], [])
    if _height_cm(boxes, cap) <= (PAGE_CM if upright else LAND_PAGE_CM):
        return "\n".join([*turn[0], rf"\begin{{{env}}}[t]", r"\centering", *sep,
                          rf"\caption{{{cap}}}", rf"\label{{{label}}}", *([size] if size else []),
                          "\n\\par\\bigskip\n".join(boxes), *tail,
                          rf"\end{{{env}}}", *turn[1], ""])
    # taller than a page: not a float, so that it breaks between its blocks. The caption and the first block are one
    # unbreakable unit; a later block that starts a page says that it continues the table.
    out = [*turn[0], CONTINUED, r"\begin{center}", *sep,
           r"\begin{minipage}{\linewidth}\centering",
           rf"\captionof{{table}}{{{cap}}}", rf"\label{{{label}}}", r"\par\medskip", *([size] if size else []), boxes[0],
           r"\end{minipage}"]
    for k, b in enumerate(boxes[1:], 2):
        out += [r"\par\bigskip", r"\begin{minipage}{\linewidth}\centering", rf"\TableContinued{{{label}}}{{{label}:{k}}}",
                *([size] if size else []), b, r"\end{minipage}"]
    return "\n".join(out + [*tail, r"\end{center}", *turn[1], ""])


PAGE_FULL_CM, LAND_FULL_CM = 23.0, 13.0     # what a page holds of a table's blocks, upright and turned (estimated)


def _pages(boxes: list, cap: str, page_cm: float, cap_chars: int) -> int:
    """Pages a table's blocks take, each block whole on a page, the caption (cap_chars a line) above the first."""
    pages, used = 1, 0.45 * (len(cap) / cap_chars + 1)
    for b in boxes:
        h = _height_cm([b], "") - 0.45
        if used and used + h > page_cm:
            pages, used = pages + 1, 0.0
        used += h
    return pages


PAGE_CM = 17.0      # a table estimated taller than this (of the 24.7 cm text height) is set to break across pages
LAND_PAGE_CM = 12.5 # the same in landscape (of the 15 cm the turned page is high)


def _height_cm(boxes: list, cap: str) -> float:
    """Estimated height of a table at the body size: rows (a wrapped row label adds lines), rules and gaps between
    blocks, and the caption."""
    h = 0.45 * (len(cap) / 90 + 1)
    for b in boxes:
        wrap = re.search(r"\\arraybackslash\}p\{([\d.]+)", b)
        lab = float(wrap.group(1)) + 0.42 if wrap else LABEL_CM
        for row in b.split(r"\\")[:-1]:
            first = _depth0_split(row.split("rule")[-1] if "rule" in row.split("&")[0] else row, "&")[0]
            h += 0.52 * (math.ceil(_cell_cm(first) / lab) if wrap else 1)
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


def required_pass(em, index, missing: list) -> None:
    for name, waits, twins in REQUIRED:
        for n in [name] + [name + t for t in twins.split()]:
            if n not in em.provenance:
                em.macro(n, r"\pending{" + waits.replace("_", r"\_") + "}", index, "not computed yet",
                         f"waits on {waits}")
                missing.append(f"{n} -- waits on {waits}")


def build(em, root: pathlib.Path, missing: list, fmt_pm, fmt_int, words) -> dict:
    data = root / "experiments/FIGS/data/v2"
    fits3, comp = _load(data / FITS), _load(data / "compute")
    fits3.update({"initbn" + m.removeprefix("init"): d for m, d in _load(data / FITS_INITBN).items()})
    # the v4 probes (class-balanced JetClass-II, v4 grid; mlst_v4.py) replace fits_v3 once pulled
    fits_v4, raw_v4 = V4.load_fits(data)
    fits = fits_v4 or fits3
    if not fits_v4:
        missing.append("v4 linear probes -- experiments/FIGS/data/v2/linprobe/fits_v4/")
    if not fits:
        missing.append(f"linear probes -- experiments/FIGS/data/v2/{FITS}/")
        return {}
    V4.design_macros(em, root, data, missing, fmt_int)
    for name, k, n_in, what in (("MlstNParamsFt", 2, 17, "two outputs (top, quark/gluon)"),
                                ("MlstNParamsFtKin", 2, 7, "two outputs, the 7 kinematic inputs"),
                                ("MlstNParamsJc", len(V4.JC1_COLS) + 1, 17, "the ten JetClass classes")):
        em.macro(name, fmt_int(V4.our_params(k, n_in)), root / "experiments/FIGS/mlst_v4.py", f"our_params({k}, {n_in})",
                 f"trainable parameters of our fine-tuned network, {what}")
    index = check_index(data)
    runs = _paper(root)
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
    fill = _load(data / FITS_FILL)
    if not fill:
        missing.append(f"substitution probes -- experiments/FIGS/data/v2/{FITS_FILL}/")
    chosen, _ = chosen_substitutions(fill, runs)
    # the released Sophon checkpoint: a row of the frozen-probe tables only, never one of the runs; pending until pulled
    sophon = _load(data / FITS_SOPHON).get(SOPHON_MODEL)
    if not sophon:
        missing.append(f"the released Sophon checkpoint's probes -- experiments/FIGS/data/v2/{FITS_SOPHON}/")
    fits_s = {**fits, SOPHON_MODEL: V4.remap(sophon)} if sophon else fits
    runs_s = {**runs, SOPHON: [SOPHON_MODEL]}
    pairs_ft = load_pairs_ft(data)
    if not pairs_ft:
        missing.append(f"fine-tuned class pairs -- experiments/FIGS/data/v2/{PAIRS_FT}/")
    # the paper's tables pool the five random partitions into one row and print no ratio column; the supplement's
    # *_full copies keep both (PI 2026-11-12)
    pooled_s = _pooled(runs_s)
    out = {"tables/mlst_probes.tex": _float("probes", table_probes_bench(fits_s, pooled_s, TAB, _paper_rows(ALL_ROWS + (INIT, SOPHON)),
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
           "tables/mlst_pairs.tex": _float("pairs", table_pairs(fits, _pooled(runs), TAB, status,
                                                                _paper_rows(rows_init),
                                                                pairs_ft, rho=False), True,
                                           fill={"__ZERO__": ""}, label_cm=PAIRS_LABEL_CM, budget=TEXT_CM * SHRINK_OK),
           "tables/mlst_pairs_full.tex": _float("pairs_full", table_pairs(fits, runs, TAB, status,
                                                                          rows_init, pairs_ft),
                                                True, fill={"__ZERO__": zero_note(fits, runs, pairs_ft)},
                                                label_cm=PAIRS_LABEL_CM, budget=TEXT_CM * SHRINK_OK),
           "tables/mlst_balance.tex": _float("balance", table_balance(fits3, fits_v4, runs, TAB, ALL_ROWS + (INIT_BN,)))}
    comparisons(em, fits, runs, status, index, fmt_pm)
    eemm_macros(em, fits, runs, index, fmt_pm, fmt_int, status)
    probe_abs_macros(em, fits, runs, index, fmt_pm)
    if fits3:
        bn_macros(em, fits3, runs, index)
        out["tables/mlst_bn.tex"] = _float("bn", table_bn(fits3, runs, TAB, ALL_ROWS))
    co = _load(data / "linprobe/fits_v4_coarse")
    if co:
        co_s = {**co, SOPHON_MODEL: sophon} if sophon else co
        # one table, the three targets side by side (PI 2026-11-12): wider than the upright page, so turned
        out["tables/mlst_coarse.tex"] = _float("coarse", table_coarse(co_s, pooled_s, TAB,
                                                                      _paper_rows(ALL_ROWS + (INIT,))), True, whole="turned", colsep_pt=2.75)
        out["tables/mlst_coarse_full.tex"] = _float("coarse_full", table_coarse(co_s, runs_s, TAB,
                                                                                ALL_ROWS + (INIT, INIT_BN, SOPHON)), True)
    else:
        missing.append("probes on the coarse targets -- experiments/FIGS/data/v2/linprobe/fits_v4_coarse/")
    hf = _load(data / "linprobe/fits_v4_qcdhf")
    if hf:
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
    jc1_ft_ovr, jc1_probe_ovr = V4.load_family(data, "jc1_ft_ovr"), _load(data / V4.ROC_DIR / "jc1_probe_ovr")
    if jc1_ft_ovr and jc1_probe_ovr:
        out["tables/mlst_jc1_class_auc.tex"] = _float("jc1_class_auc", table_jc1_class_auc(
            jc1_ft_ovr, jc1_probe_ovr, _pooled(runs), TAB, _paper_rows(ALL_ROWS)), True)

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
    if mlp:
        out["tables/mlst_mlp.tex"] = _float("mlp", table_mlp(mlp, runs, TAB, rows_init, raw_v4, mlp_nat, kin=True), True)
        mlp_macros(em, mlp, runs, index, fmt_pm)
        mlp_natural_macros(em, mlp_nat, runs, index, fmt_pm)
    else:
        missing.append(f"nonlinear probes -- experiments/FIGS/data/v2/{MLP_FITS_V4}/")
    out["tables/mlst_fill.tex"] = _float("fill", table_fill(raw_v4, fill, fits, _pooled(runs), TAB,
                                                            _paper_rows(ALL_ROWS + (INIT_BN,)), drop=False, chosen=chosen), True)
    out["tables/mlst_fill_full.tex"] = _float("fill_full", table_fill(raw_v4, fill, fits, runs, TAB, ALL_ROWS + (INIT_BN,)), True)
    out["tables/mlst_fill_main.tex"] = _float("fill_main", table_fill(raw_v4, fill, fits, runs, TAB, MAIN), True)
    fill_macros(em, raw_v4, fill, fits, runs, index, fmt_pm, missing)
    loss_share_macros(em, comp, runs, index, fmt_pm, root)
    atlas = _load(data / "head" / "atlas_res")
    if atlas:
        out["tables/mlst_mass.tex"] = _float("mass", table_mass(atlas, runs, TAB))
        # the binned values as the mass figures draw them (atlas_res_fine: ten jet-pT bins, the same jets and inclusive
        # values as atlas_res)
        fine = _load(data / "head" / "atlas_res_fine") or atlas
        out["tables/mlst_mass_detail.tex"] = _float("mass_detail", md := table_mass_detail(fine, runs, TAB), True, fill={"__BLANK__": (
            r"bins with fewer than \MlstMassMinJets{} jets are blank." if " & --" in md.split("%SPLIT%", 1)[-1]
            else r"every bin holds at least \MlstMassMinJets{} jets.")})
        mass_macros(em, atlas, runs, index, fmt_pm)
    else:
        missing.append("ATLAS mass resolution -- experiments/FIGS/data/v2/head/atlas_res/")
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
        out["tables/mlst_sculpt.tex"] = _float("sculpt", table_sculpt(sculpt, runs, TAB, SETS + (FLAV if any(
            m in sculpt for a in FLAV for m in runs[a]) else ())), True,
                                               fill={"__SCULPT_BINS__": bins, **sculpt_fill(sculpt, runs)})
        sculpt_macros(em, sculpt, sculpt_v2, runs, index, fmt_pm)
    else:
        missing.append("mass sculpting -- experiments/FIGS/data/v2/head/*_sculpting.json")
    boot_file = data / "head" / "sculpting_boot.json"
    if boot_file.exists():
        sb = json.loads(boot_file.read_text())
        out["tables/mlst_sculpt_boot.tex"] = _float("sculpt_boot", table_sculpt_boot(sb, runs, sculpt),
                                                    fill={"__REPLICAS__": fmt_int(sb["replicas"])}, whole="upright", colsep_pt=6)
    else:
        missing.append("sculpting bootstrap -- experiments/FIGS/data/v2/head/sculpting_boot.json")
    composition_macros(em, data, root, missing)
    size_macros(em, fits, fmt_int, index)
    method_macros(em, root, words)
    ft_recipe(em, data, {m[len("mtx-"):] for m in pre_runs}, fmt_int, missing, words)
    tag, per, absent = load_ft(data)
    if absent:
        missing.append("fine-tuning read-outs -- experiments/FIGS/data/v2/ft_mlst/: " + ", ".join(absent))
    if not (data / "budget" / "budget_metrics.json").exists():
        missing.append("fixed-budget fine-tuning -- experiments/FIGS/data/v2/budget/budget_metrics.json")
    boot = {d.name: _load(d) for d in sorted((data / "bootstrap").glob("*")) if d.is_dir() and "." not in d.name}
    body, worst = table_boot(boot, runs, fits, per if tag else None, balanced=bool(fits_v4))
    if fits_v4 and "probe_v4" not in boot:
        missing.append("test-sample bootstrap of the class-balanced probes -- experiments/FIGS/data/v2/bootstrap/probe_v4/")
    out["tables/mlst_boot.tex"] = _float("boot", body, True, fill={"__JC2COMP__": "class-balanced" if fits_v4
                                                                   else "natural-composition"})
    boot_macros(em, boot, worst, index)
    if tag is not None:
        # the fine-tuned training-size figures (A8, A8-part, F8): as the probes' above
        out["tables/mlst_ft.tex"] = _float("ft", table_by_size(
            "ft", "ft", lambda t, ds, met, a: size_ft(per, _pooled(runs), t, met, None if a == SCRATCH else a),
            (SCRATCH_LABEL, SCRATCH), {k: n for k, n, *_ in SUMMARY_PANELS}), True, whole="upright")
        out["tables/mlst_ft_full.tex"] = _longtable("ft_full", table_ft(per, runs, TAB, ALL_ROWS))
        # every point of the JetClass-II per-class figures (PI 2026-11-13: restored, supplement)
        out["tables/mlst_per_class.tex"] = _longtable("per_class", table_per_class(fits, per, _pooled(runs), TAB, root,
                                                                                   native_tex))
        body, fewest = table_ft_rej(per, runs, TAB, ALL_ROWS)
        out["tables/mlst_ft_rej.tex"] = _float("ft_rej", body, True,
                                               fill={"__FEWEST__": fmt_int(fewest) if fewest else "--"})
        ft_macros(em, per, runs, index, fmt_pm)
        topcheck_macros(em, data, per, index, fmt_pm, missing)
        # quark/gluon on Pythia only, Herwig apart (PI 2026-11-12); top and quark/gluon are wider than the upright
        # text block at the columns' precision, so _float sets them in landscape, no row label wrapped
        out["tables/mlst_bench_top.tex"] = _float("bench_top", V4.table_bench(
            "top", per, runs, MAIN, LABEL, SCRATCH, TAB, TAB.two, kin=(LABEL["KIN"], KIN_INIT)),
            True)
        out["tables/mlst_bench_qg.tex"] = _float("bench_qg", V4.table_bench(
            "qg", per, runs, MAIN, LABEL, SCRATCH, TAB, TAB.two), True)
        out["tables/mlst_bench_qg_herwig.tex"] = _float("bench_qg_herwig", V4.table_bench_herwig(
            per, runs, MAIN, LABEL, SCRATCH, TAB, TAB.two))
        out["tables/mlst_summary.tex"] = _float("summary", table_summary(fits_s, fill, chosen, per, runs_s, TAB), True,
                                                pages=True, whole="turned", colsep_pt=6)
        out["tables/mlst_summary_auc.tex"] = _float("summary_auc", table_summary(
            fits_s, fill, chosen, per, runs_s, TAB, SUMMARY_AUC_GRID, "summary_auc"), True, pages=True, whole="turned", colsep_pt=6)
        out["tables/mlst_labelsets.tex"] = _float("labelsets", table_labelsets(per, runs, TAB), True, pages=True, whole="turned", colsep_pt=6)
    gap_macros(em, fits, per if tag else {}, runs, index, fmt_pm)
    out["tables/mlst_per_run.tex"] = _longtable("per_run", table_per_run(fits, per if tag else {}, runs, TAB, fill,
                                                                         chosen))
    XR, xfits = xbx_section(em, root, data, index, runs, missing, out, fmt_pm, fmt_int)
    if tag is not None and atlas and xfits and sculpt:
        rows = masseffect_rows(atlas, fits_s, fill, chosen, per, mlp, XR, xfits, sculpt, _load(data / "head" / "sculpting_ft"),
                               runs)
        out["tables/mlst_masseffect.tex"] = _float("masseffect", table_masseffect(rows, runs, TAB), True, pages=True,
                                                   whole="turned", colsep_pt=6)
    t = partrepro(em, data, per if tag else {}, fmt_pm, missing, runs={**runs, "__bench__": MAIN})
    if t:
        out["tables/mlst_partrepro.tex"] = _float("partrepro", t)
    sweep = data / "budget" / "lrsweep_metrics.json"
    if sweep.exists():
        out["tables/mlst_lrsweep.tex"] = _float("lrsweep", table_lrsweep(json.loads(sweep.read_text())))
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
            pre, ref, pooled, SETS, lab, SCRATCH, 1_000_000, TAB, TAB.two, part="rej"), True, fill=jfill, whole="turned", colsep_pt=2.5)
        out["tables/mlst_jetclass_all.tex"] = _float("jetclass_all", V4.table_jetclass(
            pre, ref, pooled, SETS + FLAV + ("RAND",), lab, SCRATCH, 1_000_000, TAB, TAB.two, literature=False,
            vals=jc1_values(per, pooled)), True)
        out["tables/mlst_jetclass_nbkg.tex"] = _float("jetclass_nbkg", V4.table_jetclass_nbkg(
            pre, ref, pooled, SETS + FLAV + ("RAND",), lab, SCRATCH, 1_000_000), True)
        if tag is not None:
            out["tables/mlst_published.tex"] = _float("published", table_published(per, runs, pre, ref, TAB), True, pages=True, whole="turned", colsep_pt=6)
        jetclass_macros(em, pre, runs, src, fmt_pm, words)
        jc1_boot_macros(em, data, fmt_int, missing)
        out["tables/mlst_rand_prong.tex"] = _float("rand_prong", rp := table_rand_prong(pre, runs, root, TAB), True, fill={"__PRONG__": (
            r"which prong-count boundaries each keeps ($\bullet$: no group holds classes of both prong counts),"
            if r"\bullet" in rp else "none keeps a boundary between two, three and four prongs (for each pair of these "
            "prong counts, every partition has a group holding classes of both);")})
    else:
        missing.append("JetClass per-class curves -- experiments/FIGS/data/v2/curves/full_*.json")
    out["tables/mlst_jetclass_probe.tex"] = _float("jetclass_probe", table_jetclass_probe(fits_s, runs_s, TAB,
                                                                                          rows_init + (SOPHON,)), True)
    if comp:
        ftg, nodev = ft_gpu_types(data)
        if nodev:
            missing.append(f"GPU of {sum(nodev.values())} fixed-budget fine-tuning cells (JetClass, JetClass-II): "
                           "experiments/FT/budget_metrics.py records it for top and quark/gluon only")
        body, hours = table_compute(comp, runs, fmt_int, extra=((LABEL["KIN"], KIN_MODEL),), ft_gpus=ftg,
                                    ft_nodev=nodev)
        out["tables/compute.tex"] = body.replace("__CAPTION__", printed_caption("compute"))
        compute_macros(em, comp, runs, hours, index, fmt_int)
        if len(set(pre_runs) & set(comp)) < len(pre_runs):
            missing.append(f"compute records: {len(set(pre_runs) & set(comp))} of {len(pre_runs)} paper runs")
    required_pass(em, index, missing)
    # a table whose data is not pulled yet is a marked placeholder, never the file an older generator left behind
    for key, name, wide, fill_ in PLACEHOLDERS:
        out.setdefault(f"tables/{name}.tex", _float(key, r"\pending{no result yet}", wide, fill_))
    for key, name in LONGTABLES:
        out.setdefault(f"tables/{name}.tex", _float(key, r"\pending{no result yet}", True))
    left = [k for k, v in out.items() if V4.TOK_OPEN in v] + [n for n, b, _ in em.macros if V4.TOK_OPEN in b]
    if left:
        raise SystemExit(f"FATAL: unprinted value tokens in {left}")
    return out
