"""The MLST paper's results tables and macros, read by experiments/FIGS/make_tables.py (build()).

Every value is the mean +- standard deviation over the pretraining runs of a configuration that the
paper reports (configs/arms/v2_paper_runs.txt), PDG-rounded (make_tables.fmt_pm). A configuration
with runs still missing prints the runs it has and a red marker "n of N runs". Configurations with
two runs print both values (a standard deviation from two runs has one degree of freedom); a lone
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
  linprobe/fits_v3_mlp2/, ..._initbn/      nonlinear probes (top, quark/gluon, JetClass; JetClass-II
                                           until fits_v4_mlp is pulled)
  head/atlas_res/<model>.json              ATLAS mass resolution and median response
  head/curves_v2/<model>_sculpting_v2.json, head/<model>_sculpting.json   QCD mass sculpting
  compute/<run>.json                       per epoch: GPU and seconds
  ft_mlst/*.json, budget/*.json            fine-tuning read-outs; curves/full_*.json  JetClass per class
  bootstrap/<family>/*.json                test-sample bootstrap; partrepro/<variant>.json  ParT on top
  xbx/<signal>/<model>.json, xbx_scratch/<signal>/s<k>.json   X->bb and X->bs
  jc2bal/census.json, jc2_test/counts.json class counts; index.json  {path: sha256}

Outputs (paper/journal/tables/): main text T2 mlst_jetclass, T3 mlst_bench_top, T4 mlst_bench_qg,
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

import numpy as np

_V4_SPEC = importlib.util.spec_from_file_location("mlst_v4", pathlib.Path(__file__).resolve().parent / "mlst_v4.py")
V4 = importlib.util.module_from_spec(_V4_SPEC)
_V4_SPEC.loader.exec_module(V4)

MAIN = ("L188", "L162", "R42_Q1", "R16_Q1")
MASS = ("L162_MASS", "R16_Q1_MASS")
RAND = tuple(f"RAND2_p{k}" for k in range(1, 6))
FLAV = ("FLAV_F0", "FLAV_F1", "FLAV_F1R")
ALL_ROWS = MAIN + MASS + RAND + FLAV
INIT = "INIT"
INIT_MODELS = ("init-s1", "init-s2", "init-s3")
# the randomly initialized networks with their BatchNorm statistics recomputed on the training jets
# of the best70 twins (experiments/MTX/bn_twins_v2.py --init): checkpoint init_bn
INIT_BN = "INIT_BN"
INIT_BN_MODELS = ("initbn-s1", "initbn-s2", "initbn-s3")
# the 188-class model trained on the four-momenta only (plan 3.7): one run, not in the grid file
KIN_MODEL = "mtx-l188kin-s1"
KIN_INIT = KIN_MODEL[len("mtx-"):]
PRIMARY = "best70"


def _ck(model: str) -> str:
    """The checkpoint a model is reported at."""
    return "init" if model.startswith("init-") else "init_bn" if model.startswith("initbn-") else PRIMARY


LABEL = {"L188": r"\Kfull{} classes", "L162": r"\Kres{} classes", "R42_Q1": r"\Kmid{} classes",
         "R16_Q1": r"\Kcoarse{} classes", "L162_MASS": r"\Kres{} classes, mass regression",
         "R16_Q1_MASS": r"\Kcoarse{} classes, mass regression", "FLAV_F0": "random partition, no flavor boundary",
         "FLAV_F1": r"random partition, $b$-quark boundary",
         "FLAV_F1R": "random partition, random flavor boundary", INIT: "randomly initialized network",
         INIT_BN: "randomly initialized network, BatchNorm recomputed",
         "KIN": r"\Kfull{} classes, kinematic inputs only",
         **{f"RAND2_p{k}": f"random partition {k}" for k in range(1, 6)}}
FITS, MLP_FITS = "linprobe/fits_v3", "linprobe/fits_v3_mlp"
MLP_FITS_V4, FITS_INITBN, MLP_FITS_INITBN = ("linprobe/fits_v3_mlp2", "linprobe/fits_v3_initbn",
                                             "linprobe/fits_v3_mlp2_initbn")
FITS_FILL, MLP_BAL = "linprobe/fits_v4_fill", "linprobe/fits_v4_mlp"
# (dataset key, metric, column heading)
TASKS = (("jc2_188", "macro_auc_ovr", "JetClass-II"), ("jc2_188", "macro_auc_resonant", "resonant"),
         ("jc2_188", "macro_auc_qcd", "QCD"), ("jc1", "macro_auc_ovr", "JetClass"),
         ("top", "auc", "top"), ("qg", "auc", "q/g"), ("qg_herwig", "auc", "q/g, Herwig"))
TASKS_N = (TASKS[0], ("jc2_188", "macro_auc_trained", "JetClass-II, classes present in training"),
           TASKS[2], TASKS[3], TASKS[4], TASKS[5])
# the JetClass-II tasks, the only ones a ratio of 1 - AUC is taken on (PI, 2026-11-02)
JC2_TASKS = (TASKS[0], TASKS_N[1], TASKS[1], TASKS[2])
PAIRS = (("bc_vs_bq_cs", r"$X\to bc$ vs $bq$, $cs$", ("label_X_bc",), ("label_X_bq", "label_X_cs")),
         ("bb_vs_cc", r"$X\to b\bar b$ vs $c\bar c$", ("label_X_bb",), ("label_X_cc",)),
         ("bbqq_vs_ccqq", r"$X\to YY\to bbqq$ vs $ccqq$", ("label_X_YY_bbqq",), ("label_X_YY_ccqq",)),
         ("ee_vs_mm", r"$X\to ee$ vs $\mu\mu$", ("label_X_ee",), ("label_X_mm",)),
         ("qcd_bb_vs_cc", r"QCD $bb$ vs $cc$", ("label_QCD_bb",), ("label_QCD_cc",)),
         ("qcd_b_vs_light", r"QCD with $b$ vs light", "qcd_with_b", ("label_QCD_light",)),
         ("qcd_c_vs_light", r"QCD with $c$, no $b$, vs light", "qcd_with_c_no_b", ("label_QCD_light",)))
PAIR_DS, PAIR_N = "jc2_188_pairs", "1000000"
MLP_TASKS = (("jc2_188", "JetClass-II", "macro_auc_ovr"), ("jc2_188_res", "resonant", "macro_auc_resonant"),
             ("jc2_188_qcd", "QCD", "macro_auc_qcd"), ("jc1", "JetClass", "macro_auc_ovr"),
             ("top", "top", "auc"), ("qg", "q/g", "auc"), ("qg_herwig", "q/g, Herwig", "auc"),
             ("bc_vs_bq_cs", r"$X\to bc$ vs $bq$, $cs$", "auc"), ("bb_vs_cc", r"$X\to b\bar b$ vs $c\bar c$", "auc"),
             ("bbqq_vs_ccqq", r"$X\to YY\to bbqq$ vs $ccqq$", "auc"), ("qcd_bb_vs_cc", r"QCD $bb$ vs $cc$", "auc"),
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
# the zero-fill control's two masks (fillcontrol/<mask>/); the paper no longer reads them, the mock-up
# figures (mockup_figures.figure_f5) still do
MASKS = ("TopReference", "EnergyFlowQG")
# Fine-tuning: (task, read-out file, data set inside it, metric, column heading)
FT_TAGS = ("best70",)
FT_TASKS = (("jc2", "leg188_leg1_metrics", None, "macro_auc_ovr", "JetClass-II"),
            ("jc2_qcd", "leg188_leg1_metrics", None, "macro_auc_qcd", "JetClass-II, QCD classes"),
            ("jc1", "leg2_metrics", None, "macro_auc_ovr", "JetClass"),
            ("top", "bench_metrics", "top", "auc", "top"),
            ("qg", "bench_metrics", "qg", "auc", "q/g"),
            ("qg_herwig", "bench_metrics_herwig", "qg", "auc", "q/g, Herwig"))
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
    """Two runs: both values, to the decimal that shows their difference to two digits, but never
    beyond four significant figures."""
    d = abs(a - b)
    if d == 0:
        return _lone(a) + sep + _lone(b)
    cap = 3 - math.floor(math.log10(max(abs(a), abs(b))))
    dec = max(0, min(1 - math.floor(math.log10(d)), cap))
    return sep.join(_sep(f"{x:.{dec}f}") for x in (a, b))


def _cell(values: list, n_expected: int, fmt_pm, scale: float = 1.0, sep: str = ", ") -> str:
    """mean +- SD over three or more runs present; two runs print both values; a lone run
    prints alone; missing runs add a red marker; a lower limit among them a geq sign."""
    flag = any(isinstance(x, Unconverged) for x in values)
    bound = any(isinstance(x, Bound) for x in values)
    v = [x * scale for x in values if x is not None]
    if not v:
        return r"\pending{no run yet}"
    if len(v) > 2:
        body = fmt_pm(float(np.mean(v)), float(np.std(v, ddof=1)))
    elif len(v) == 2:
        body = _two(v[0], v[1], sep)
    else:
        body = _lone(v[0])
    if bound:
        body = r"$\geq$" + body
    if flag:
        body += UNCONVERGED
    if len(v) < n_expected:
        body += rf" \pending{{{len(v)} of {n_expected}}}"
    return body


def _pct_cell(values: list, n_expected: int, fmt_pm) -> str:
    """_cell of values in per cent, the per-cent sign before any marker."""
    b = _cell(values, n_expected, fmt_pm)
    if b.startswith(r"\pending"):
        return b
    head, sp, tail = b.partition(r" \pending")
    return head + r"\%" + (sp + tail if sp else "")


def _row(arm: str, runs: dict) -> str:
    """The row label; a random partition, pretrained with a single seed, also names that seed."""
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
    run over the 188-class run of its seed); the randomly initialized networks have no such ratio."""
    if arm in (INIT, INIT_BN):
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


def table_probes(fits, runs, fmt_pm, rows, full_count=None, fill=None, chosen=None) -> str:
    """A1, linear probes at the largest training set: JetClass-II accuracy and AUC averaged over all
    classes, the classes at the full test count, the resonant and the QCD classes, with the ratio of
    1 - AUC to the 188-class run as a labelled column; JetClass accuracy and AUC; top, quark/gluon
    and Herwig: accuracy, AUC, R50 and R30 under the substitution chosen on validation (chosen, from
    chosen_substitutions; zero, marked in the heading, while it is pending)."""
    fill, chosen = fill or {}, chosen or {}
    n2 = _largest(fits, "jc2_188")
    out = [r"\begin{tabular}{l" + "c" * 8 + "}", r"\toprule",
           r" & \multicolumn{6}{c}{JetClass-II} & \multicolumn{2}{c}{JetClass} \\",
           r"\cmidrule(lr){2-7}\cmidrule(lr){8-9}",
           r"pretraining & accuracy & AUC & AUC, full count & AUC, resonant & AUC, QCD & "
           r"$\rho$ & accuracy & AUC \\", r"\midrule"]
    for arm in rows:
        ms, n = runs[arm], len(runs[arm])
        r2 = [_probe_rec(fits, m, "jc2_188", n2) for m in ms]
        r1 = [_probe_rec(fits, m, "jc1") for m in ms]
        ratio, nr = _ratio_to_full(fits, runs, arm, lambda m: _oma(_probe(fits, m, "jc2_188", n2, "macro_auc_ovr")))
        cells = [_cell([None if r is None else r["accuracy"] for r in r2], n, fmt_pm),
                 _cell([_probe(fits, m, "jc2_188", n2, "macro_auc_ovr") for m in ms], n, fmt_pm),
                 _cell([_auc_over(r, full_count) for r in r2], n, fmt_pm) if full_count else "--",
                 _cell([_probe(fits, m, "jc2_188", n2, "macro_auc_resonant") for m in ms], n, fmt_pm),
                 _cell([_probe(fits, m, "jc2_188", n2, "macro_auc_qcd") for m in ms], n, fmt_pm),
                 _cell(ratio, nr, fmt_pm) if nr else "--",
                 _cell([None if r is None else r["accuracy"] for r in r1], n, fmt_pm),
                 _cell([None if r is None else r["macro_auc_ovr"] for r in r1], n, fmt_pm)]
        out.append(_row(arm, runs) + " & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", "%SPLIT%", r"\begin{tabular}{l" + "c" * 12 + "}", r"\toprule",
            " & " + " & ".join(rf"\multicolumn{{4}}{{c}}{{{h}}}" for h in ("top", "quark/gluon", "quark/gluon, Herwig"))
            + r" \\",
            " & " + " & ".join(rf"\multicolumn{{4}}{{c}}{{{_subst_head(chosen, ds)}}}" for ds in ("top", "qg", "qg_herwig"))
            + r" \\",
            r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-13}",
            "pretraining & " + " & ".join(h for _, h in BENCH4 * 3) + r" \\", r"\midrule"]
    for arm in rows:
        ms, n = runs[arm], len(runs[arm])
        cells = []
        for ds in ("top", "qg", "qg_herwig"):
            recs = [_bench_rec(fits, fill, chosen, m, ds) for m in ms]
            for k, _ in BENCH4:
                cells.append(_cell([_rej(r, k) if k.startswith("r") else (None if r is None else r[k]) for r in recs],
                                   n, fmt_pm))
        out.append(_row(arm, runs) + " & " + " & ".join(cells) + r" \\")
    return "\n".join(out + [r"\bottomrule", r"\end{tabular}", ""])


def table_probes_n(fits, runs, fmt_pm, rows, fill=None, chosen=None) -> str:
    """A1, linear probes against the training-set size: JetClass-II AUC, JetClass accuracy, R30 on
    top, quark/gluon and Herwig (under the substitution chosen on validation, as table_probes)."""
    fill, chosen = fill or {}, chosen or {}
    out = []
    for ds, met, title in (("jc2_188", "macro_auc_ovr", "JetClass-II, AUC"), ("jc1", "accuracy", "JetClass, accuracy"),
                           ("top", "r30", r"top, $R_{30}$"), ("qg", "r30", r"quark/gluon, $R_{30}$"),
                           ("qg_herwig", "r30", r"quark/gluon, Herwig, $R_{30}$")):
        bench = ds in SUBST_KEY
        sizes = (_bench_sizes(fits, fill, chosen, ds) if bench else
                 sorted({n for f in fits.values() for ck in f["cells"].values() for n in ck.get(ds, {})}, key=int))
        if not sizes:
            continue
        if bench:
            title += f" ({_subst_head(chosen, ds)})"
        out += [r"\begin{tabular}{l" + "c" * len(sizes) + "}", r"\toprule",
                rf"{title} & " + " & ".join(f"{int(n):,}".replace(",", "{,}") for n in sizes) + r" \\", r"\midrule"]
        for arm in rows:
            vals = []
            for n in sizes:
                recs = [_bench_rec(fits, fill, chosen, m, ds, n) if bench else _probe_rec(fits, m, ds, n)
                        for m in runs[arm]]
                vals.append(_cell([_rej(r, met) if met == "r30" else (None if r is None else r[met]) for r in recs],
                                  len(runs[arm]), fmt_pm))
            out.append(LABEL[arm] + " & " + " & ".join(vals) + r" \\")
        out += [r"\bottomrule", r"\end{tabular}", "%SPLIT%"]
    return "\n".join(out[:-1])


def _pair(fits, m, key):
    c = fits.get(m, {}).get("cells", {}).get(_ck(m), {}).get(PAIR_DS, {}).get(PAIR_N, {}).get(key)
    return None if c is None else c["auc"]


def separated(fits, runs, key) -> set:
    """Run indices whose 188-class probe separates the pair's test jets completely (1 - AUC = 0): no
    ratio of 1 - AUC to that run exists."""
    return {_idx(m) for m in runs["L188"] if _pair(fits, m, key) == 1}


def table_pairs(fits, runs, fmt_pm, status, rows) -> str:
    """A11, binary probes between JetClass-II classes: the AUC, and the ratio of 1 - AUC to the
    188-class run as a labelled column ('--' where that run separates the test jets completely);
    a dagger marks a pair the label set merges."""
    mark = {"merged": r"$^{\dagger}$", "partly": r"$^{\ddagger}$", "apart": ""}
    out = []
    for part in (PAIRS[:4], PAIRS[4:]):
        out += [r"\begin{tabular}{l" + "cc" * len(part) + "}", r"\toprule",
                "pretraining & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{h}}}" for _, h, _, _ in part) + r" \\",
                "".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(part))),
                " & " + " & ".join([r"AUC & $\rho$"] * len(part)) + r" \\",
                r"\midrule"]
        for arm in rows:
            cells = []
            for key, _, _, _ in part:
                cells.append(_cell([_pair(fits, m, key) for m in runs[arm]], len(runs[arm]), fmt_pm)
                             + mark[status.get(arm, {}).get(key, "apart")])
                r, n = _ratio_to_full(fits, runs, arm, lambda m: _oma(_pair(fits, m, key)))
                sep = separated(fits, runs, key) & {_idx(m) for m in runs[arm]}
                cells.append(_cell(r, n, fmt_pm) if n and not sep else "--")
            out.append(_row(arm, runs) + " & " + " & ".join(cells) + r" \\")
        out += [r"\bottomrule", r"\end{tabular}", "%SPLIT%"]
    return "\n".join(out[:-1])


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
        zero = lambda ref, arms: {_idx(m) for m in runs[ref] if f(m) == 0} & {_idx(m) for a in arms for m in runs[a]}
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
    probes only: the class-balanced fits score best70 alone."""
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


def table_bn(fits3, runs, fmt_pm, rows) -> str:
    """A6: JetClass-II (natural composition, largest training set) as trained and with the BatchNorm
    statistics recomputed: the AUC, and the ratio of 1 - AUC to the 188-class run."""
    n = _largest(fits3, "jc2_188")
    out = [r"\begin{tabular}{lcccc}", r"\toprule",
           r" & \multicolumn{2}{c}{AUC} & \multicolumn{2}{c}{$\rho$} \\",
           r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
           r"pretraining & as trained & recomputed & as trained & recomputed \\", r"\midrule"]

    def auc(m, ck):
        c = fits3.get(m, {}).get("cells", {}).get(ck, {}).get("jc2_188", {}).get(n)
        return None if c is None else c["macro_auc_ovr"]
    for arm in rows:
        ms = runs[arm]
        cells = [_cell([auc(m, PRIMARY) for m in ms], len(ms), fmt_pm), _cell([auc(m, "best70_bn") for m in ms], len(ms), fmt_pm)]
        for ck in (PRIMARY, "best70_bn"):
            r, k = _ratio_to_full(fits3, runs, arm, lambda m: _oma(auc(m, ck)))
            cells.append(_cell(r, k, fmt_pm) if k else "--")
        out.append(_row(arm, runs) + " & " + " & ".join(cells) + r" \\")
    a = [_cell([auc(m, "init") for m in INIT_MODELS], 3, fmt_pm),
         _cell([auc(m, "init_bn") for m in INIT_BN_MODELS], 3, fmt_pm)]
    out.append(LABEL[INIT] + " & " + " & ".join(a) + r" & -- & -- \\")
    return "\n".join(out + [r"\bottomrule", r"\end{tabular}", ""])


def table_balance(fits3, fits4, runs, fmt_pm, rows) -> str:
    """A13: JetClass-II at 10^6 training jets, natural composition (fits_v3) beside class-balanced
    (fits_v4): AUC averaged over the classes and the ratio of 1 - AUC to the 188-class run."""
    out = [r"\begin{tabular}{lcccc}", r"\toprule",
           r" & \multicolumn{2}{c}{natural composition} & \multicolumn{2}{c}{class-balanced} \\",
           r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
           r"pretraining & AUC & $\rho$ & AUC & "
           r"$\rho$ \\", r"\midrule"]
    for arm in rows:
        cells = []
        for f in (fits3, fits4):
            if not f:
                cells += [r"\pending{fits}"] * 2
                continue
            v = lambda m, f=f: _probe(f, m, "jc2_188", PAIR_N, "macro_auc_ovr")
            cells.append(_cell([v(m) for m in runs[arm]], len(runs[arm]), fmt_pm))
            r, k = _ratio_to_full(f, runs, arm, lambda m, v=v: _oma(v(m)))
            cells.append(_cell(r, k, fmt_pm) if k else "--")
        out.append(_row(arm, runs) + " & " + " & ".join(cells) + r" \\")
    return "\n".join(out + [r"\bottomrule", r"\end{tabular}", ""])


# ------------------------------------------------------------------ nonlinear probes (A5)
def load_mlp(data: pathlib.Path) -> dict:
    """The nonlinear probes: top, quark/gluon, Herwig and JetClass from fits_v3_mlp2 (their tasks do
    not depend on the JetClass-II composition); JetClass-II and its pairs from fits_v4_mlp once
    pulled (class-balanced, as the linear probes), else from fits_v3_mlp2. JetClass-II values of the
    two versions are never mixed."""
    mlp = {**_load(data / MLP_FITS), **_load(data / MLP_FITS_V4)}
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
            mlp.setdefault(model, {"cells": {}})["cells"].setdefault(ck, {}).update(v)
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


def table_mlp(mlp, runs, fmt_pm, rows, raw_v4=None) -> str:
    """A5: the nonlinear probe (one hidden layer, three seeds averaged) at the largest training set:
    JetClass-II AUC (all, resonant, QCD) with the ratio of 1 - AUC to the 188-class run; JetClass AUC;
    top, quark/gluon and Herwig AUC and R30; the class pairs' AUC and ratio; and the linear probe on
    the head's 512-unit hidden layer beside the one on the class token."""
    out = [r"\begin{tabular}{l" + "c" * 11 + "}", r"\toprule",
           r" & \multicolumn{4}{c}{JetClass-II} & JetClass & \multicolumn{2}{c}{top} & \multicolumn{2}{c}{quark/gluon}"
           r" & \multicolumn{2}{c}{Herwig} \\",
           r"\cmidrule(lr){2-5}\cmidrule(lr){7-8}\cmidrule(lr){9-10}\cmidrule(lr){11-12}",
           r"pretraining & AUC & resonant & QCD & $\rho$ & AUC & AUC & "
           r"$R_{30}$ & AUC & $R_{30}$ & AUC & $R_{30}$ \\", r"\midrule"]
    for arm in rows:
        ms, n = runs[arm], len(runs[arm])
        r, k = _ratio_to_full(None, runs, arm, lambda m: _mlp(mlp, m, "jc2_188"))
        cells = [_cell([_mlp_val(mlp, m, t) for m in ms], n, fmt_pm) for t in ("jc2_188", "jc2_188_res", "jc2_188_qcd")]
        cells.append(_cell(r, k, fmt_pm) if k else "--")
        cells.append(_cell([_mlp_val(mlp, m, "jc1") for m in ms], n, fmt_pm))
        for t in ("top", "qg", "qg_herwig"):
            cells += [_cell([_mlp_val(mlp, m, t) for m in ms], n, fmt_pm),
                      _cell([_mlp_val(mlp, m, t, "r30") for m in ms], n, fmt_pm)]
        out.append(_row(arm, runs) + " & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    for part in (MLP_PAIRS[:3], MLP_PAIRS[3:]):
        out += ["%SPLIT%", r"\begin{tabular}{l" + "cc" * len(part) + "}", r"\toprule",
                "pretraining & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{h}}}" for _, h, _, _ in part) + r" \\",
                " & " + " & ".join([r"AUC & $\rho$"] * len(part)) + r" \\",
                r"\midrule"]
        for arm in rows:
            cells = []
            for key, _, _, _ in part:
                cells.append(_cell([_mlp_val(mlp, m, key) for m in runs[arm]], len(runs[arm]), fmt_pm))
                r, k = _ratio_to_full(None, runs, arm, lambda m, key=key: _mlp(mlp, m, key))
                cells.append(_cell(r, k, fmt_pm) if k else "--")
            out.append(_row(arm, runs) + " & " + " & ".join(cells) + r" \\")
        out += [r"\bottomrule", r"\end{tabular}"]
    hid = [m for m in runs["L188"] if "jc2bal.hidden" in (raw_v4 or {}).get(m, {}).get("cells", {}).get(PRIMARY, {})]
    if hid:
        def cell(m, key, met):
            c = raw_v4[m]["cells"][PRIMARY].get(key, {})
            c = c[max(c, key=int)] if c else None
            return None if c is None else (_rej(c, met) if met.startswith("r") else c[met])
        rows_h = (("class token", (("jc2bal", "macro_auc_ovr"), ("qg", "auc"), ("qg", "r30"))),
                  ("512-unit hidden layer", (("jc2bal.hidden", "macro_auc_ovr"), ("qg.hidden", "auc"), ("qg.hidden", "r30"))))
        out += ["%SPLIT%", r"\begin{tabular}{lccc}", r"\toprule",
                r"\Kfull{} classes, linear probe on & JetClass-II AUC & quark/gluon AUC & quark/gluon $R_{30}$ \\",
                r"\midrule"]
        for name, keys in rows_h:
            out.append(name + " & " + " & ".join(_cell([cell(m, k, met) for m in hid], len(runs["L188"]), fmt_pm)
                                                 for k, met in keys) + r" \\")
        out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out + [""])


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
    The kinematic-only model has nothing missing and is read from fits."""
    s = chosen.get(ds)
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


def table_fill(raw_v4, fill, fits, runs, fmt_pm, rows) -> str:
    """A16: every configuration with the missing inputs substituted by zero, the training mean or
    random draws. JetClass-II (largest training set): the relative drop in accuracy with the slots the
    top and the quark/gluon data lack substituted. Top and quark/gluon (largest training set): R30 of
    the linear probe under each substitution; the kinematic-only model has nothing missing."""
    out = [r"\begin{tabular}{l" + "c" * 6 + "}", r"\toprule",
           r" & \multicolumn{3}{c}{JetClass-II, top slots} & \multicolumn{3}{c}{JetClass-II, quark/gluon slots} \\",
           r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}", "pretraining & " + " & ".join(SUBST * 2) + r" \\", r"\midrule"]
    for arm in rows:
        ms = runs[arm]
        out.append(_row(arm, runs) + " & " + " & ".join(
            _pct_cell([_jc2_drop(raw_v4, fill, m, t, s) for m in ms], len(ms), fmt_pm)
            for t in ("top", "qg") for s in SUBST) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", "%SPLIT%", r"\begin{tabular}{l" + "c" * 6 + "}", r"\toprule",
            r" & \multicolumn{3}{c}{top, $R_{30}$} & \multicolumn{3}{c}{quark/gluon, $R_{30}$} \\",
            r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}", "pretraining & " + " & ".join(SUBST * 2) + r" \\", r"\midrule"]
    for arm in rows:
        ms = runs[arm]
        out.append(_row(arm, runs) + " & " + " & ".join(
            _cell([_fill_r30(fill, m, t, s) for m in ms], len(ms), fmt_pm) for t in ("top", "qg") for s in SUBST) + r" \\")
    kin = _rej(_probe_rec(fits, KIN_MODEL, "top"), "r30")
    out.append(LABEL["KIN"] + " & " + (_lone(kin) if kin is not None else r"\pending{run}")
               + r" & \multicolumn{2}{c}{nothing missing} & -- & -- & -- \\")
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
MASS_ROWS = (INIT,) + MAIN + MASS


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
    """T5: the jet mass from frozen features, an MLP regression per configuration, against the groomed
    (soft-drop) mass: ATLAS's resolution and the median response over all test jets (FIGURE_PLAN T5:
    the overall value; resonant and QCD jets separately in A17)."""
    g = MASS_GROUPS[0][0]
    out = [r"\begin{tabular}{lcc}", r"\toprule", r" & resolution & median response \\", r"\midrule"]
    out.append("groomed mass & " + " & ".join(
        _lone(v) if (v := _groomed(atlas, g, k)) is not None else "--" for k in ("resolution", "median")) + r" \\")
    out.append(r"\midrule \multicolumn{3}{l}{MLP regression on frozen features} \\")
    for arm in MASS_ROWS:
        ms = runs[arm]
        out.append(LABEL[arm] + " & " + " & ".join(_cell([_ares(atlas, m, "mlp", g, k) for m in ms], len(ms), fmt_pm)
                                                   for k in ("resolution", "median")) + r" \\")
    return "\n".join(out + [r"\bottomrule", r"\end{tabular}", ""])


def _binned_one(atlas, axis: str) -> list | None:
    """The bin edges of an axis, one list across every file and estimator, or the build stops."""
    e = {json.dumps(g[axis]["edges"]) for d in atlas.values() for est in d["estimators"].values()
         for g in est.values() if axis in g}
    if len(e) > 1:
        raise SystemExit(f"FATAL: the atlas_res files bin {axis} differently")
    return json.loads(e.pop()) if e else None


def table_mass_detail(atlas, runs, fmt_pm) -> str:
    """A17: the inclusive values of every estimator (groomed, the regression output of the mass
    models, the frozen-feature ridge and MLP regressions), and the resolution and median response of
    the groomed mass and the regression output against jet pT and generator mass."""
    ng = len(MASS_GROUPS)
    out = [r"\begin{tabular}{ll" + "c" * 2 * ng + "}", r"\toprule",
           rf" & & \multicolumn{{{ng}}}{{c}}{{resolution}} & \multicolumn{{{ng}}}{{c}}{{median response}} \\",
           rf"\cmidrule(lr){{3-{2 + ng}}}\cmidrule(lr){{{3 + ng}-{2 + 2 * ng}}}",
           "estimator & pretraining & " + " & ".join(h for _ in range(2) for _, _, h in MASS_GROUPS) + r" \\",
           r"\midrule"]
    for est, name, arms in (("regressed", "regression output", MASS), ("ridge", "ridge, frozen features", MASS_ROWS),
                            ("mlp", "MLP, frozen features", MASS_ROWS)):
        for arm in arms:
            ms = runs[arm]
            out.append(f"{name} & {LABEL[arm]} & " + " & ".join(
                _cell([_ares(atlas, m, est, g, k) for m in ms], len(ms), fmt_pm)
                for k in ("resolution", "median") for g, _, _ in MASS_GROUPS) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    for axis, title in (("vs_pt", r"jet $p_{\mathrm{T}}$ [GeV]"), ("vs_mgen", r"generator soft-drop mass [GeV]")):
        edges = _binned_one(atlas, axis)
        if not edges:
            continue
        bins = [f"{a:g}--{b:g}" for a, b in zip(edges, edges[1:])]
        for key, kname in (("resolution", "resolution"), ("median", "median response")):
            out += ["%SPLIT%", r"\begin{tabular}{ll" + "c" * len(bins) + "}", r"\toprule",
                    rf"{kname} & {title} & " + " & ".join(bins) + r" \\", r"\midrule"]
            for g, _, gh in MASS_GROUPS:
                gv = _groomed(atlas, g, key, axis) or []
                out.append(f"groomed mass & {gh} & "
                           + " & ".join("--" if x is None else _lone(x) for x in gv) + r" \\")
                for arm in MASS:
                    ms = runs[arm]
                    per = [_ares(atlas, m, "regressed", g, key, axis) for m in ms]
                    out.append(f"{LABEL[arm]} & {gh} & " + " & ".join(
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
    """A14: Jensen-Shannon distance (x 10^2) between the soft-drop mass distributions of all QCD test
    jets and of those passing a cut on 1 - p_QCD at the stated QCD efficiency, and below the runs the
    distance a random selection of as many QCD jets shows."""
    lines = [r"\begin{tabular}{lcc}", r"\toprule",
             r"pretraining & " + " & ".join(rf"$\epsB = {100 * float(e):g}\%$" for e in SCULPT_EFFS)
             + r" \\", r"\midrule"]
    for arm in rows:
        lines.append(LABEL[arm] + " & " + " & ".join(
            _cell([_sculpt(sculpt, m, e) for m in runs[arm]], len(runs[arm]), fmt_pm, 1e2) for e in SCULPT_EFFS) + r" \\")
    if sculpt:
        lines += [r"\midrule", "random selection of as many QCD jets & "
                  + " & ".join(_lone(1e2 * _sculpt_floor_distance(sculpt, e)) for e in SCULPT_EFFS) + r" \\"]
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", ""])


def _curve(sculpt_v2: dict, m: str) -> dict:
    """{efficiency: (distance, floor distance)} of a run's sculpting curve (curves_v2)."""
    d = sculpt_v2.get(m)
    if d is None:
        return {}
    k = int(np.count_nonzero(d["hist_before"]))
    return {round(p["eff"], 9): (p["js_distance"], _floor_distance(k, p["n_pass"], d.get("n_qcd"))) for p in d["curve"]}


def sculpt_macros(em, sculpt, sculpt_v2, runs, index, fmt_pm) -> None:
    """Distances at the two working points (MlstSculpt<arm><Ten|One>), the random-selection floor,
    run-by-run ratios between configurations and to the floor, and on the stored efficiency grid the
    window where the 17-class model sculpts more than the 188-class one (the paired ratio's mean minus
    twice its SD above 1) and, at the lowest stored efficiency, the largest excess over the floor in
    units of the floor's SD."""
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
        for k in ("MlstSculptDiffEffLo", "MlstSculptDiffEffHi", "MlstSculptDiffLoEdge", "MlstSculptLowestEff",
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
    em.macro("MlstSculptDiffEffLo", pc(min(diff)) if diff else "none", index,
             "lowest stored QCD efficiency at which the 17/188 distance ratio exceeds 1 by twice its SD")
    em.macro("MlstSculptDiffEffHi", pc(max(diff)) if diff else "none", index, "highest such efficiency")
    em.macro("MlstSculptDiffLoEdge", ("at" if min(diff) == effs[0] else "above") + " the lowest stored efficiency"
             if diff else "none", index, "where the window's lower edge lies: at the lowest stored QCD efficiency "
             "(the 17/188 difference holds at the tightest cut stored) or above it")
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
    return tag, merge_budget(per, data / "budget" / "budget_metrics.json"), absent


# 2026-10-31 (plan 3.1, 3.4): below the full set every fine-tuning is the fixed-budget recipe
# (experiments/FT/budget_metrics.py), which supersedes the earlier cells there; JetClass-II is the
# class-balanced task at every size. (budget tree, sub-record or None) per task; the full sets.
BUDGET_TREE = {"top": ("leg_top", None), "qg": ("leg_qg", None), "qg_herwig": ("leg_qg", "herwig"),
               "jc1": ("leg2", None), "jc2": ("leg_jc2b", None), "jc2_qcd": ("leg_jc2b", None)}
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
    return device.replace("NVIDIA ", "").replace("GeForce ", "").replace("Tesla ", "").replace("-SXM2-32GB", "")


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
    return ("1" + (rf" \pending{{{have} of {n}}}" if have < n else "")) if have else r"\pending{no run yet}"


def table_ft(per, runs, fmt_pm, rows) -> str:
    """A1, fine-tuning against the training-set size, in each task's metric (FT_METRIC): the AUC
    averaged over the classes on class-balanced JetClass-II (and over its QCD classes), accuracy on
    JetClass, R30 on top, quark/gluon and Herwig; the network trained from scratch last. The
    JetClass-II block adds one column: the ratio of 1 - AUC at FT_RATIO_N jets to the 188-class run."""
    out = []
    for key, _, _, _, title in FT_TASKS:
        met, mname = FT_METRIC[key]
        sizes = _sizes(per, key)
        if not sizes:
            out += [r"\begin{tabular}{lc}", r"\toprule", rf"{title} & \pending{{not yet read out}} \\",
                    r"\bottomrule", r"\end{tabular}", "%SPLIT%"]
            continue
        ratio = key == "jc2"          # the only fine-tuned ratio of 1 - AUC in the paper (ROC_INTERFACE 3.3)
        out += [r"\begin{tabular}{l" + "c" * (len(sizes) + ratio) + "}", r"\toprule",
                rf"{title}, {mname} & " + " & ".join(f"{n:,}".replace(",", "{,}") for n in sizes)
                + (r" & $\rho$, $10^{6}$ jets" if ratio else "") + r" \\",
                r"\midrule"]
        for arm in _ft_rows(per, key, runs, rows):
            cells = [_cell([_ftv(per, key, m, n, met) for m in runs[arm]], len(runs[arm]), fmt_pm) for n in sizes]
            if ratio:
                cells.append(_ft_ratio_cell(per, runs, arm, fmt_pm))
            out.append(_row(arm, runs) + " & " + " & ".join(cells) + r" \\")
        out.append("from scratch & " + " & ".join(
            [(_cell(v, len(v), fmt_pm) if (v := _scratch(per, key, n, met)) else "--") for n in sizes]
            + ["--"] * ratio) + r" \\")
        out += [r"\bottomrule", r"\end{tabular}", "%SPLIT%"]
    return "\n".join(out[:-1])


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
             "pretraining & " + " & ".join(rf"\multicolumn{{4}}{{c}}{{{h}}}" for h in ("top", "quark/gluon", "quark/gluon, Herwig"))
             + r" \\", "".join(rf"\cmidrule(lr){{{2 + 4 * i}-{5 + 4 * i}}}" for i in range(len(tasks))),
             " & " + " & ".join(h for _, h in BENCH4 * len(tasks)) + r" \\", r"\midrule"]
    arms = [a for a in _ft_rows(per, "top", runs, rows)]
    for arm in arms + [SCRATCH, "KIN"]:
        cells = []
        for key in tasks:
            if not _sizes(per, key):
                cells += [r"\pending{read-out}"] * 4
                continue
            n = _sizes(per, key)[-1]
            if arm == SCRATCH:
                recs = per[key].get(SCRATCH, {}).get(n, [])
                cells += [_cell([val(r, k) for r in recs], len(recs), fmt_pm) if recs else "--" for k, _ in BENCH4]
            elif arm == "KIN":
                recs = per[key].get(KIN_INIT, {}).get(n, []) if key == "top" else []
                cells += [_cell([val(r, k) for r in recs], 1, fmt_pm) if recs else "--" for k, _ in BENCH4]
            else:
                recs = [(per[key].get(m[len("mtx-"):], {}).get(n) or [None])[0] for m in runs[arm]]
                cells += [_cell([None if r is None else val(r, k) for r in recs], len(runs[arm]), fmt_pm) for k, _ in BENCH4]
        name = "from scratch" if arm == SCRATCH else LABEL["KIN"] if arm == "KIN" else _row(arm, runs)
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
    b = data / "budget" / "budget_metrics.json"
    if b.exists():
        for k, c in json.loads(b.read_text())["cells"].items():
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


def table_jetclass_probe(fits, runs, fmt_pm, rows) -> str:
    """A1, JetClass linear probes in the form of ParT's Table 1: accuracy, AUC averaged over the classes
    (one-vs-one where the fit has it) and each signal class's rejection of QCD at ParT's working point."""
    idx = {c: i for i, c in enumerate(V4.LPV_CLASSES)}
    head = [r"\begin{tabular}{l" + "c" * (2 + len(V4.JC1_COLS)) + "}", r"\toprule",
            r" & & & \multicolumn{" + str(len(V4.JC1_COLS)) + r"}{c}{rejection of QCD} \\",
            r"\cmidrule(lr){4-" + str(3 + len(V4.JC1_COLS)) + "}",
            r"pretraining & accuracy & AUC & " + " & ".join(h for _, h in V4.JC1_COLS) + r" \\", r"\midrule"]
    out = list(head)
    for arm in rows:
        recs = [_probe_rec(fits, m, "jc1") for m in runs[arm]]
        cells = [_cell([None if r is None else r["accuracy"] for r in recs], len(recs), fmt_pm),
                 _cell([None if r is None else r.get("macro_auc_ovo", r["macro_auc_ovr"]) for r in recs], len(recs), fmt_pm)]
        for c, _ in V4.JC1_COLS:
            w = [None if r is None else r["rejection_vs_qcd"].get(str(idx[c])) for r in recs]
            mark = r"$^\dagger$" if any(x is not None and x["n_bkg_pass"] < V4.JC1_MIN_BKG for x in w) else ""
            cells.append(_cell([None if x is None else (Bound(x["rejection"]) if x.get("is_bound") else x["rejection"])
                                for x in w], len(recs), fmt_pm) + mark)
        out.append(_row(arm, runs) + " & " + " & ".join(cells) + r" \\")
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
    "change": {c: pp}}. drop_c = mean over indices of 100 (AUC_c^188 - AUC_c^17); change_c = max over
    the 162-, 43- and 17-class label sets of |mean over indices of 100 (AUC_c^K - AUC_c^188)|.
    Candidates: classes with at least THIN_TEST class-balanced test jets (census["counts"]["ft_test"];
    every class when census is None). hurt = the NHURT largest drop_c; stable = the NSTABLE smallest
    change_c among the rest. Each list in F3's row order (_class_rows, resonant before QCD)."""
    mean = {a: {c: float(np.mean(v)) for c, v in per_class_auc_diff(fits, runs, a).items() if v}
            for a in ("L162", "R42_Q1", "R16_Q1")}
    drop = {c: -v for c, v in mean["R16_Q1"].items()}
    change = {c: max(abs(mean[a][c]) for a in mean) for c in drop if all(c in mean[a] for a in mean)}
    counts = census["counts"]["ft_test"] if census else None
    order = {int(r["jet_label"]): i for i, r in enumerate(_class_rows(root))}
    cand = sorted((c for c in change if counts is None or counts[c] >= THIN_TEST), key=order.get)
    hurt = sorted(cand, key=lambda c: -drop[c])[:NHURT]
    stable = sorted((c for c in cand if c not in hurt), key=lambda c: change[c])[:NSTABLE]
    return {"hurt": sorted(hurt, key=order.get), "stable": sorted(stable, key=order.get), "drop": drop, "change": change}


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
        em.macro("MlstRocJcTwoStableMaxChange", _sig2(max(sel["change"][c] for c in sel["stable"])), index, src,
                 "largest |AUC change| at any label set of the stable classes, pp")


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


def roc_macros(em, roc: dict, runs: dict, index, missing: list, fmt_int) -> None:
    """From the ROC curves (mlst_v4.load_roc): MlstRocCut{pair}{W} and MlstRocCutJcOne{class}Full (_roc_cut
    over a configuration's runs, against the family's grid; load_roc requires one grid per
    family), MlstRocNBkg{pair} and MlstRocRejMax{pair} (the test set's background jets and that over
    JC1_MIN_BKG, where every curve stops), MlstRocNoCurveEemm (the configurations of F4 that draw no
    X->ee vs X->mumu curve, by F4's rule: _roc_cut 'not reached', i.e. some run has fewer than
    ROC_MIN_POINTS grid points with JC1_MIN_BKG background jets; a noun phrase). The all-runs-null rule of ROC_INTERFACE 3.2 was
    replaced by the figure's rule (2026-11-02, to be flagged to the PI)."""
    pairs = roc.get("pairs", {})
    grid = next(iter(pairs.values()))["eps_grid"] if pairs else None
    for key in ROC_PAIRS:
        p = PAIR_WORD[key]
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


def eemm_macros(em, fits: dict, runs: dict, index, fmt_pm, fmt_int) -> None:
    """The X->ee vs X->mumu linear probe (fits_v4_sib, merged by mlst_v4.load_fits): MlstLpEemmAuc{W},
    and its jets, MlstPairEemm{NTrain,NVal,NTestPerSide} (one value when every model agrees, else the range)."""
    for arm in MAIN + FLAV + (INIT, INIT_BN):
        v = [_pair(fits, m, "ee_vs_mm") for m in runs[arm]]
        if any(x is not None for x in v):
            em.macro(f"MlstLpEemmAuc{WORD[arm]}", _cell(v, len(v), fmt_pm, sep=TEXT_SEP), index,
                     f"linear probe ee_vs_mm (fits_v4_sib): AUC, {arm}")
    recs = [c for m, d in fits.items()
            if (c := d["cells"].get(_ck(m), {}).get(PAIR_DS, {}).get(PAIR_N, {}).get("ee_vs_mm"))]
    if not recs:
        return
    span = lambda xs: fmt_int(min(xs)) if min(xs) == max(xs) else f"{fmt_int(min(xs))}--{fmt_int(max(xs))}"
    em.macro("MlstPairEemmNTrain", span([c["n_train"] for c in recs]), index, "fits_v4_sib ee_vs_mm: n_train")
    em.macro("MlstPairEemmNVal", span([c["n_val"] for c in recs]), index, "fits_v4_sib ee_vs_mm: n_val")
    em.macro("MlstPairEemmNTestPerSide", span([c[k] for c in recs for k in ("n_test_signal", "n_test_background")]),
             index, "fits_v4_sib ee_vs_mm: n_test_signal and n_test_background")


def table_per_class(fits, runs, fmt_pm, root, native_tex) -> str:
    """A12: per class of JetClass-II (class-balanced test set, largest training set), the AUC of the
    188-class model and the ratio of 1 - AUC to it of every configuration of F3."""
    cols = ("L162", "R42_Q1", "R16_Q1") + RAND + FLAV
    rat = {a: per_class_ratios(fits, runs, a) for a in cols}
    full = [_per_class(fits, m) for m in runs["L188"]]
    head = (r"class & AUC, \Kfull{} & \Kres{} & \Kmid{} & \Kcoarse{} & "
            + " & ".join(f"p{k}" for k in range(1, 6)) + r" & no flavor & $b$ quark & random flavor \\")
    rows, last = [], None
    for r in _class_rows(root):
        c = int(r["jet_label"])
        if last is not None and r["R16_Q1"] != last:
            rows.append(r"\midrule")
        last = r["R16_Q1"]
        aucs = [f[c] for f in full if f is not None and c in f]
        cells = [_cell(aucs, len(full), fmt_pm)] + [_cell(rat[a].get(c, []), max(len(runs[a]), 1), fmt_pm) for a in cols]
        rows.append(native_tex(r["class_name"]) + " & " + " & ".join(cells) + r" \\")
    return "\n".join(["{\\scriptsize", "\\setlength{\\tabcolsep}{2pt}", r"\begin{longtable}{l" + "c" * (1 + len(cols)) + "}",
                      r"\caption{__CAPTION__}\label{tab:perclass}\\", r"\toprule", head, r"\midrule", r"\endfirsthead",
                      r"\toprule", head, r"\midrule", r"\endhead", r"\bottomrule", r"\endlastfoot", *rows,
                      r"\end{longtable}", "}", ""])


def per_class_macros(em, fits, runs, root, index, fmt_pm, native_tex) -> None:
    """Where the cost sits (F3): for each coarser level, the classes whose 1 - AUC at least doubles,
    the class with the largest ratio, and for the classes merged with a decay that differs only in
    lepton flavor and only in quark flavor, their number, median ratio and share of the summed
    ln ratio over the resonant classes (the per-class and the aggregate cost); in the random partitions,
    the median ratio of the classes merged with a flavor counterpart over that of those kept apart."""
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
        for kind, key_of, kw in (("lepton", lep, "Lepton"), ("quark", q, "Quark")):
            mc = merged_with(arm, key_of) & set(res)
            v = [res[c] for c in mc]
            em.macro(f"MlstPerClass{w}{kw}N", str(len(v)), index,
                     f"resonant classes sharing their {arm} group with a decay that differs only in {kind} flavor (count)")
            med = [float(np.median(v))] if v else []
            _emit_ratio(em, f"MlstPerClass{w}{kw}Median", med, 1, fmt_pm, index,
                        f"median over those classes of the mean per-class ratio, {arm} over 188", "errors",
                        body=_lone(med[0]) if med else "--")
            em.macro(f"MlstPerClass{w}{kw}Share", f"{100 * sum(math.log(res[c]) for c in mc) / tot:.0f}\\%" if v and tot
                     else "--", index, f"their share of the summed ln ratio over the resonant classes, {arm}")
    for kind, key_of, kw in (("quark", q, "Flav"), ("lepton", lep, "Lepton")):
        vals = []
        for arm in RAND:
            rat = per_class_ratios(fits, runs, arm)
            mean = {c: float(np.mean(v)) for c, v in rat.items() if v}
            sib = {by_name[a] for a in key_of if sum(key_of[b] == key_of[a] for b in key_of) > 1}
            mc = merged_with(arm, key_of)
            mv = [mean[c] for c in sib & mc if c in mean]
            av = [mean[c] for c in sib - mc if c in mean]
            vals.append(float(np.median(mv)) / float(np.median(av)) if mv and av else None)
        _emit_ratio(em, f"MlstPerClassRand{kw}MergedOverApart", vals, len(vals), fmt_pm, index,
                    f"random partitions: median per-class ratio of the classes merged with a decay that differs only in "
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
    out = [r"\begin{tabular}{l" + "c" * (len(pairs) + 1 + len(V4.JC1_COLS)) + "}", r"\toprule",
           r" & \multicolumn{3}{c}{prong-count boundary kept} & & \multicolumn{" + str(len(V4.JC1_COLS))
           + r"}{c}{fine-tuned rejection over the \Kfull-class model} \\",
           r"\cmidrule(lr){2-4}\cmidrule(lr){6-" + str(5 + len(V4.JC1_COLS)) + "}",
           r"partition & 2/3 & 2/4 & 3/4 & mixed groups & " + " & ".join(h for _, h in V4.JC1_COLS) + r" \\", r"\midrule"]
    for arm in RAND:
        g = collections.defaultdict(set)
        for r in rmap:
            if r["class_name"] in prong:
                g[r[arm]].add(prong[r["class_name"]])
        kept = ["$\\bullet$" if not any(a in s and b in s for s in g.values()) else "---" for a, b in pairs]
        mixed = sum(len(s) > 1 for s in g.values())
        m = runs[arm][0]
        ref = next((x for x in runs["L188"] if _idx(x) == _idx(m)), None)
        cells = []
        for c, _ in V4.JC1_COLS:
            a, b = _jc1_rej(pre, m, c), _jc1_rej(pre, ref, c) if ref else None
            cells.append(_lone(a["rejection"] / b["rejection"]) + (r"$^\dagger$" if min(a["n_bkg_pass"], b["n_bkg_pass"])
                                                                  < V4.JC1_MIN_BKG else "") if a and b else "--")
        out.append(_row(arm, runs) + " & " + " & ".join(kept) + f" & {mixed} & " + " & ".join(cells) + r" \\")
    return "\n".join(out + [r"\bottomrule", r"\end{tabular}", ""])


# ------------------------------------------------------------------ test-sample uncertainty (A7)
# JetClass-II columns: the bootstrap of the test jets (bootstrap/<family>/<task>_N<N>.json) of the same
# composition as the fits the paper's ratios read: the class-balanced probe_v4 family
# (scripts/build_curve_jobs.py boot_probe_v4_body) when fits_v4 is in use, the natural-composition probe
# family only while the ratios still come from fits_v3; never one for the other. Top and
# quark/gluon columns: R30's counting uncertainty, 1/sqrt(passing background jets), which the fits and
# read-outs store as r30_rel_stat (the bootstrap scores 1 - AUC only, experiments/STATS/paired_bootstrap.py).
BOOT_COLS = (("JetClass-II", "probe", {True: (("probe_v4", "jc2bal"),), False: (("probe", "jc2_188"),)}, "macro"),
             ("top", "probe", "top", "r30"), ("q/g", "probe", "qg", "r30"),
             ("JetClass-II", "ft", (("jc2bal_ft", "jc2bal"),), "macro"),
             ("top", "ft", "top", "r30"), ("q/g", "ft", "qg", "r30"))
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
             r" & \multicolumn{3}{c}{linear probe} & \multicolumn{3}{c}{fine-tuning} \\",
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
        lines.append(LABEL[arm] + " / " + LABEL[ref] + " & " + " & ".join(cells) + r" \\")
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
PARTREPRO = (("part", "ParT's script, unchanged"), ("head", r"\quad + Sophon's output head"),
             ("rescale", r"\quad + Sophon's jet-$p_{\mathrm{T}}$ rescaling"))


def partrepro(em, data: pathlib.Path, per, fmt_pm, missing: list) -> str | None:
    """A9 and its macros: ParT's top-tagging script on our cluster, then with Sophon's output head and
    with the head and Sophon's rescaling of the four-momenta (cumulative, one run each), beside ParT's
    published row and our network trained from scratch."""
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
    lit = next(r for r in json.loads(V4.LIT.read_text())["top"] if r["method"] == "ParT")
    out = [r"\begin{tabular}{lccccc}", r"\toprule",
           r" & accuracy & AUC & $R_{50}$ & $R_{30}$ & background jets passing at $R_{30}$ \\", r"\midrule"]
    for v, name in PARTREPRO:
        d = docs.get(v)
        out.append(name + " & " + (" & ".join([f"{d['accuracy']:.4f}", f"{d['auc']:.4f}", _sep(str(round(d['r50']))),
                                              _sep(str(round(d['r30']))), str(d['r30_n_bkg_pass'])])
                                   if d else r"\multicolumn{5}{c}{\pending{run}}") + r" \\")
    pm = lambda s: str(s).replace(" ± ", r"\,$\pm$\,")
    out.append(rf"ParT, published~\cite{{part}} & {lit['acc']} & {lit['auc']} & {pm(lit['r50'])} & {pm(lit['r30'])} & -- \\")
    if per and _sizes(per, "top"):
        n = _sizes(per, "top")[-1]
        recs = per["top"].get(SCRATCH, {}).get(n, [])
        if recs:
            out.append(r"\midrule our network from scratch & " + " & ".join(
                _cell([r[k] for r in recs], len(recs), fmt_pm) for k in ("accuracy", "auc", "r50", "r30")) + r" & -- \\")
    return "\n".join(out + [r"\bottomrule", r"\end{tabular}", ""])


def partrepro_note() -> str:
    """T3's footnote (FIGURE_PLAN T3): where the gap between our from-scratch network and ParT's
    published row comes from, in the A9 macros; ParT's published R30 from literature_benchmarks.json."""
    lit = next(r for r in json.loads(V4.LIT.read_text())["top"] if r["method"] == "ParT")
    return (r"ParT's top-tagging training script, unchanged, on our cluster gives $R_{30}$ of \MlstPartreproRthirtyPart{} "
            "(published " + V4._lit_cell(lit["r30"]) + r"); with Sophon's output head, \MlstPartreproRthirtyHead{}; "
            r"with the head and Sophon's jet-$p_{\mathrm{T}}$ rescaling, \MlstPartreproRthirtyRescale{} (one run each; "
            r"our network from scratch, which uses the rescaling, \MlstFtTopRthirtyScratch{}); "
            r"table~\ref{tab:partrepro}.")


def table_lrsweep(doc: dict) -> str:
    """A4: the learning-rate sweep at 10^4 top and quark/gluon jets: each rate's best validation
    accuracy per task and their sum; the rate with the largest sum is chosen (mlst_v4.choose_rates)."""
    pre, scr = V4.choose_rates(doc)
    by = collections.defaultdict(dict)
    for key, m in doc["cells"].items():
        tree, leaf = key.split("/")[:2]
        init, lr = leaf.rsplit("_lr", 1)
        by[("from scratch" if init == "scratch" else "pretrained, backbone", lr)][tree] = m["early_stop"]["best_metric"]
    out = [r"\begin{tabular}{llccc}", r"\toprule", r"network & learning rate & top & quark/gluon & sum \\", r"\midrule"]
    for (kind, lr), v in sorted(by.items(), key=lambda x: (x[0][0] != "pretrained, backbone", float(x[0][1]))):
        chosen = (kind != "from scratch" and lr == pre) or (kind == "from scratch" and lr == scr)
        vals = [v.get("leg_top"), v.get("leg_qg")]
        cells = ["--" if x is None else f"{x:.4f}" for x in vals]
        s = f"{sum(vals):.3f}" if None not in vals else "--"
        out.append(f"{kind} & {_power(lr)} & " + " & ".join(cells) + f" & {s}" + (r"$^{\ast}$" if chosen else "") + r" \\")
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
    training particles), and what the mean and the random-draw substitutions put in a missing slot
    (experiments/EVAL/input_variants.py)."""
    top = _features(root / "configs/finetune/TopReference.yaml")
    qg = _features(root / "configs/finetune/EnergyFlowQG.yaml")
    flags = {"part_isChargedHadron", "part_isNeutralHadron", "part_isPhoton", "part_isElectron", "part_isMuon"}
    disp = {"part_d0", "part_d0err", "part_dz", "part_dzerr"}
    out = [r"\begin{tabular}{lccccc}", r"\toprule",
           r"input & top & quark/gluon & zero after normalization & mean & random draws \\", r"\midrule"]
    for ft, fq in zip(top, qg):
        name = ft.replace("part_zero_", "part_")
        missing_t, missing_q = ft.startswith("part_zero_"), fq.startswith("part_zero_")
        z = "--"
        if missing_t or missing_q:
            z = _lone(offsets[name]) if offsets and name in offsets else r"\pending{offsets}"
        mean = "running mean of the input normalization" if missing_t or missing_q else "--"
        draw = ("a species drawn from JetClass-II's frequencies (top)" if name in flags else
                "drawn given the species (top)" if name == "part_charge" else
                "a JetClass-II particle of the same species" if name in disp else "--")
        mt, mq = ("---" if missing_t else BULLET), ("---" if missing_q else BULLET)
        out.append(f"{FEATURE_TEX.get(name, name)} & {mt} & {mq} & {z} & {mean} & {draw}" + r" \\")
    return "\n".join(out + [r"\bottomrule", r"\end{tabular}", ""])


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


def table_xbx_rej(xfits, scr, runs, fmt_pm) -> str:
    """A15: rejection at 50 % and 30 % signal efficiency in the mass window, every configuration."""
    cols = (("xbb", "head"), ("xbb", "zero_shot_xbb"), ("xbs", "head"), ("xbs", "zero_shot_xbq"))
    out = [r"\begin{tabular}{l" + "cc" * len(cols) + "}", r"\toprule",
           r" & \multicolumn{2}{c}{$X\to b\bar b$, transfer} & \multicolumn{2}{c}{$X\to b\bar b$, direct} & "
           r"\multicolumn{2}{c}{$X\to bs$, transfer} & \multicolumn{2}{c}{$X\to bs$, closest node} \\",
           "".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(cols))),
           "pretraining & " + " & ".join([r"$R_{50}$ & $R_{30}$"] * len(cols)) + r" \\", r"\midrule"]
    for arm in MAIN + MASS + RAND + FLAV + (INIT,):
        ms = runs[arm]
        cells = []
        for sig, name in cols:
            nm = "linear" if arm == INIT and name == "head" else name
            cells += [_cell([_xr30(xfits, m, sig, nm, k) for m in ms], len(ms), fmt_pm) for k in ("r50", "r30")]
        out.append(_row(arm, runs) + " & " + " & ".join(cells) + r" \\")
    s = scr.get("xbs", [])
    out.append("from scratch & -- & -- & -- & -- & " + " & ".join(
        _cell([x.get(k) for x in s], max(len(s), 1), fmt_pm) for k in ("r50", "r30")) + r" & -- & -- \\")
    return "\n".join(out + [r"\bottomrule", r"\end{tabular}", ""])


XBX_MACROS = ("XbxMx", "XbxWindowLo", "XbxWindowHi") + tuple(
    f"{s}{k}" for s in ("Xbb", "Xbs") for k in ("NTrainPerClass", "Repetition", "NEvalSignal", "NEvalQcd"))


def xbx_section(em, root, data, index, runs, missing, out, fmt_pm, fmt_int) -> None:
    """X->bb and X->bs from the pulled fits through experiments/XBS/xbx_report.py, and the rejection
    ratios and table of A15; red markers until they exist."""
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
            SimpleNamespace(LABEL=LABEL, _cell=_cell, RAND=RAND, INIT_MODELS=INIT_MODELS), MT, runs, xfits))
        out["tables/mlst_xbx_rej.tex"] = _float("xbx_rej", table_xbx_rej(xfits, scr, runs, fmt_pm), True)
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


# ------------------------------------------------------------------ compute (appendix, Data availability)
def table_compute(comp, runs, fmt_int, extra=(), ft_gpus=None, ft_nodev=None) -> tuple[str, dict]:
    """Per pretraining run: hours on each GPU type (completed epochs, training and validation passes,
    as each epoch recorded its device), their sum, and the GPU types its fine-tuning cells ran on
    (their manifests record the device, not the hours), with a marker counting its cells whose
    read-out records no device (ft_nodev). extra: (label, model) rows beside the grid."""
    lines = [r"\begin{tabular}{lllr}", r"\toprule", r"run & GPU types (hours) & GPU-hours & fine-tuning GPU types \\",
             r"\midrule"]
    hours = {}
    items = [(arm, f"{LABEL[arm]}, seed {_idx(m)}", m) for arm, ms in runs.items() if arm not in (INIT, INIT_BN) for m in ms]
    items += [(None, lab, m) for lab, m in extra]
    for arm, name, m in items:
        ftg = ", ".join((ft_gpus or {}).get(m[len("mtx-"):], [])) or "--"
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
        types = ", ".join(f"{g} ({h[g]:.0f})" for g in sorted(h, key=lambda g: -h[g]))
        lines.append(rf"{name} & {types} & {fmt_int(round(sum(h.values())))} & {ftg} \\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", ""]), hours


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
    AUC, JetClass AUC, R30 on top and quark/gluon under the substitution chosen on validation, zero
    while it is pending) and fine-tuning (JetClass-II AUC, JetClass accuracy, R30 on top,
    quark/gluon and Herwig)."""
    fill, chosen = fill or {}, chosen or {}
    head = (r"run & \multicolumn{4}{c}{linear probe} & \multicolumn{5}{c}{fine-tuning} \\" "\n"
            r" & JetClass-II AUC & JetClass AUC & top $R_{30}$ & q/g $R_{30}$ & JetClass-II AUC & JetClass accuracy & "
            r"top $R_{30}$ & q/g $R_{30}$ & Herwig $R_{30}$ \\")
    one = lambda x: "--" if x is None else _lone(x) if abs(x) >= 1 else f"{x:.4f}"

    def ft(key, m, met):
        if not per or not _sizes(per, key):
            return None
        return _ftv(per, key, m, _sizes(per, key)[-1], met)
    rows = []
    items = [(f"{LABEL[a]}, seed {_idx(m)}", m) for a in ALL_ROWS for m in runs[a]] + [(LABEL["KIN"], KIN_MODEL)]
    for name, m in items:
        p2, p1 = _probe_rec(fits, m, "jc2_188", PAIR_N), _probe_rec(fits, m, "jc1")
        pt, pq = _bench_rec(fits, fill, chosen, m, "top"), _bench_rec(fits, fill, chosen, m, "qg")
        cells = [one(None if p2 is None else p2["macro_auc_ovr"]), one(None if p1 is None else p1["macro_auc_ovr"]),
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
            rows.append(f"from scratch, seed {s + 1} & -- & -- & -- & -- & " + " & ".join(vals) + r" \\")
    return "\n".join(["{\\scriptsize", "\\setlength{\\tabcolsep}{2pt}", r"\begin{longtable}{l" + "c" * 9 + "}",
                      r"\caption{__CAPTION__}\label{tab:perrun}\\", r"\toprule", head, r"\midrule", r"\endfirsthead",
                      r"\toprule", head, r"\midrule", r"\endhead", r"\bottomrule", r"\endlastfoot", *rows,
                      r"\end{longtable}", "}", ""])


# ------------------------------------------------------------------ captions and assembly
SPREAD = ("Mean $\\pm$ standard deviation over the \\MlstNRunsWord{} pretraining runs of each level of the class "
          "hierarchy; configurations with \\RandFlavRuns{} runs give both values, and a random partition has a "
          "single run.")
SPREAD3 = "Mean $\\pm$ standard deviation over the \\MlstNRunsWord{} pretraining runs."
SPREAD_MASS = (SPREAD3[:-1] + "; the models with the mass regression have \\MlstMassNRunsWord{} runs, and both "
               "values are given.")
RATIO_COL = ("The ratio column is $\\rho$ of equation~(\\ref{eq:ratio}), $(1-\\mathrm{AUC})$ over that of the "
             "\\Kfull-class run of the same seed (a random partition: of its seed).")
AUC_DEF = "AUC: area under the curve of signal efficiency against background efficiency."
# the pairs caption's note where a 188-class run separates a pair's test jets completely (tab:pairs '--')
ZERO_NOTE = (" A dash in a ratio column: the \\Kfull-class run of that seed separates the test jets completely "
             "($1-\\mathrm{AUC} = 0$), so no ratio exists.")


def zero_note(fits, runs) -> str:
    """The tab:pairs caption's ZERO_NOTE when a 188-class run separates some pair's test jets completely."""
    return ZERO_NOTE if any(separated(fits, runs, k) for k, *_ in PAIRS) else ""


CAPTIONS = {
    "probes": ("tab:probes", "Linear probes on the frozen features at the largest training set of each task. "
               "Upper part: JetClass-II (class-balanced test set) accuracy and AUC averaged over its \\Kfull{} classes "
               "each against the rest, over the classes with the full \\JcBalTestPerClass{} test jets, over the resonant "
               "and over the QCD classes; JetClass accuracy and AUC. Lower part: top tagging and quark/gluon "
               "discrimination (also on Herwig jets) with the missing inputs substituted as chosen on validation for "
               "every model (\\LpSubstitutionTop{} for top, \\LpSubstitutionQg{} for quark/gluon; column headings): "
               "accuracy, AUC and the background rejection $1/\\epsB$ at 50\\% and 30\\% signal efficiency. "
               + RATIO_COL + " " + SPREAD
               + " " + AUC_DEF + " $^{*}$: a probe that did not converge."),
    "probes_n": ("tab:probesn", "Linear probes against the number of training jets (column headers): JetClass-II AUC "
                 "averaged over the classes, JetClass accuracy, and $R_{30}$ on top, quark/gluon and Herwig jets with "
                 "the missing inputs substituted as chosen on validation (\\LpSubstitutionTop{} for top, "
                 "\\LpSubstitutionQg{} for quark/gluon). " + SPREAD3 + " $^{*}$: a probe that did not converge."),
    "pairs": ("tab:pairs", "Binary linear probes between pairs of JetClass-II classes: the AUC and the ratio $\\rho$ "
              "of $1-\\mathrm{AUC}$, equation~(\\ref{eq:ratio}), to the \\Kfull-class run of the same seed. $^{\\dagger}$: the label set merges the two "
              "sides into one class; $^{\\ddagger}$: it merges the first side with part of the second. " + SPREAD
              + " The $b$-quark boundary and the random flavor boundary lie only in the four-quark decays; every label "
              "set but the \\Kfull-class one merges all QCD classes. The $X\\to ee$ vs $\\mu\\mu$ probe uses the jets of the "
              "two classes in the class-balanced sets (\\MlstPairEemmNTrain{} training jets, \\MlstPairEemmNTestPerSide{} "
              "test jets per side).__ZERO__"),
    "mlp": ("tab:mlp", "Nonlinear probes (a multilayer perceptron with one hidden layer of \\LpMlpHidden{} units, "
            "\\LpMlpSeedsWord{} seeds averaged) at the largest training set: JetClass-II AUC (all, resonant and QCD "
            "classes) with the ratio $\\rho$ of $1-\\mathrm{AUC}$, equation~(\\ref{eq:ratio}), to the \\Kfull-class run, JetClass AUC, and AUC and $R_{30}$ "
            "on top, quark/gluon and Herwig jets (missing inputs set to zero); then the class pairs of "
            "table~\\ref{tab:pairs}; last, the linear probe on the 512-unit hidden layer of the \\Kfull-class models' "
            "pretraining head beside the one on the class token. " + SPREAD),
    "bn": ("tab:bn", "Linear probes on JetClass-II (natural composition, largest training set) with the BatchNorm "
           "statistics as accumulated in pretraining and recomputed on training jets: the AUC averaged over the classes "
           "and the ratio $\\rho$ of $1-\\mathrm{AUC}$, equation~(\\ref{eq:ratio}), to the \\Kfull-class run. The class-balanced probes use the checkpoints "
           "as trained, so this check uses the natural-composition test set. " + SPREAD),
    "balance": ("tab:balance", "JetClass-II linear probes with $10^{6}$ training jets on the natural-composition test "
                "set and on the class-balanced one: AUC averaged over the classes and the ratio $\\rho$ of $1-\\mathrm{AUC}$, equation~(\\ref{eq:ratio}), to "
                "the \\Kfull-class run. " + SPREAD),
    "fill": ("tab:fill", "The three substitutions of the missing inputs (zero, the training mean, random draws of a "
             "real particle) for every configuration, at the largest training set. Upper part: relative drop in "
             "JetClass-II accuracy (per cent) when the slots the top (\\FillTopNZero{}) or the quark/gluon "
             "(\\FillQgNZero{}) data lack are substituted, against the same model with every input. Lower part: "
             "$R_{30}$ of the linear probe on top and quark/gluon jets under each substitution; the model with "
             "kinematic inputs only has nothing missing. The substitution chosen on validation is "
             "\\LpSubstitutionTop{} for top and \\LpSubstitutionQg{} for quark/gluon. " + SPREAD),
    "mass": ("tab:mass", "Jet mass from frozen features: a multilayer perceptron regresses the mass from the frozen "
             "features of each configuration; the groomed (soft-drop) mass is the reference. Resolution: $\\sigma_r$ of "
             "equation~(\\ref{eq:resolution}), half the 15.9--84.1\\% inter-quantile range of the response "
             "$r = m_{\\mathrm{est}}/\\mgen$ divided by its median; "
             "and the median response, over all JetClass-II test jets with a generator match (resonant and QCD jets "
             "separately in table~\\ref{tab:massdetail}); lower resolution is better. The regression target is "
             "centred on the training median of each of the \\Kfull{} classes, a truth-label input the groomed-mass "
             "row does not have. A multilayer perceptron is used because the mass is not linear in the features "
             "(ridge regression in table~\\ref{tab:massdetail}). The randomly initialized network shows what a "
             "network that never trained exposes. The mass regression in pretraining has the weight "
             "$\\lambda = \\MlstMassLambda{}$ at both label sets, a share of the training loss of "
             "\\MlstMassLossShareResMass\\% (\\Kres{} classes) and \\MlstMassLossShareCoarseMass\\% (\\Kcoarse{} "
             "classes). " + SPREAD_MASS),
    "mass_detail": ("tab:massdetail", "Jet-mass regression in detail, over all test jets and over resonant and QCD "
                    "jets. Upper part: resolution and median response of the "
                    "regression output of the models pretrained with the mass regression, and of ridge and multilayer "
                    "perceptron regressions on frozen features. Lower parts: resolution and median response of the "
                    "groomed mass and of the regression output against jet $p_{\\mathrm{T}}$ and against the generator "
                    "soft-drop mass; bins with fewer than \\MlstMassMinJets{} jets are blank. " + SPREAD_MASS),
    "sculpt": ("tab:sculpt", "Mass sculpting: Jensen--Shannon distance (base 2) $\\times10^{2}$ between the "
               "$m_{\\mathrm{SD}}$ distributions (__SCULPT_BINS__) of all QCD test jets and of those passing a cut on "
               "$1-g_{\\mathrm{QCD}}$ at the QCD efficiency $\\epsB$ given; lower is less sculpting. The last row is the "
               "distance expected for a random selection of as many of the QCD test jets (drawn without replacement). "
               + SPREAD_MASS),
    "ft": ("tab:ft", "Fine-tuning against the number of training jets (column headers), in each task's metric: AUC "
           "averaged over the classes on class-balanced JetClass-II (all classes and the QCD classes), accuracy on "
           "JetClass, and $R_{30}$ on top, quark/gluon and Herwig jets. " + SPREAD + " Each pretrained run is "
           "fine-tuned once; from scratch: \\MlstFtRefSeedsWord{} training seeds, on the same jets for top and "
           "quark/gluon and on JetClass each on its own training subset. The last column of "
           "the JetClass-II block is $\\rho$ of equation~(\\ref{eq:ratio}) at $10^{6}$ training jets, $1-\\mathrm{AUC}$ "
           "over that of the \\Kfull-class run of the same seed."),
    "ft_rej": ("tab:ftrej", "Fine-tuning on the full training sets of top and quark/gluon (also scored on Herwig "
               "jets), every configuration: accuracy, AUC and $1/\\epsB$ at 50\\% and 30\\% signal efficiency. "
               + SPREAD + " Every rejection rests on at least __FEWEST__ background jets passing the cut."),
    "bench_top": ("tab:benchtop", "Top tagging, fine-tuned on the full training set (\\LitPartTopNTrainMillion{} million "
                  "jets): accuracy, AUC and background rejection $1/\\epsB$ at 50\\% and 30\\% signal efficiency. "
                  "Published rows from the tables cited; f.t.\\ marks a model pretrained on JetClass and fine-tuned. Our "
                  "pretrained models see the \\FillTopNZero{} inputs the top data lack set to zero; the model with "
                  "kinematic inputs only, a diagnostic of that mismatch, has none missing (one run). Uncertainties: "
                  "ours, the standard deviation over the pretraining runs, one fine-tuning each, and over three seeds "
                  "from scratch; ParT, the spread over \\LitPartFtHeadReinits{} fine-tunings of one pretrained model "
                  "with different output-layer initializations, quoted for the run of median accuracy; L-GATr and "
                  "OmniLearned, the standard deviation of five trainings. Bold: the best value of each column, and "
                  "every value within one standard deviation of it."),
    "bench_qg": ("tab:benchqg", "Quark/gluon discrimination with the particle type as five flags and the charge "
                 "(ParT's exp inputs), fine-tuned on the full training set (\\LitQgTrainMillion{} million Pythia jets) "
                 "and scored on Pythia and on Herwig jets: accuracy, AUC and $1/\\epsB$ at 50\\% and 30\\% quark "
                 "efficiency. Our models see the \\FillQgNZero{} displacement inputs set to zero. Published rows (exp "
                 "inputs) from the tables cited; PFN-Ex's uncertainty is half the interquartile range of ten trainings, "
                 "the others as in table~\\ref{tab:benchtop}, with nine trainings for OmniLearned. Bold as in "
                 "table~\\ref{tab:benchtop}."),
    "boot": ("tab:boot", "Uncertainty of the label-set comparisons at the largest training set, in per cent: the "
             "standard deviation of the logarithm of the ratio over the pretraining runs, then that from the finite "
             "test sample. JetClass-II: the ratio of $1-\\mathrm{AUC}$, the test-sample part from a bootstrap of the "
             "__JC2COMP__ test jets (\\MlstBootResamples{} resamples within each class). Top and quark/gluon: the ratio of "
             "$R_{30}$, the test-sample part from the number of background jets passing the cut."),
    "jetclass": ("tab:jetclass", "JetClass, in the form of ParT's Table~1: accuracy, AUC averaged over all pairs of "
                 "classes (one against one, ParT's convention; on JetClass's equal class sizes it equals the "
                 "one-against-rest average, checked), and for each signal class the rejection of QCD jets at 50\\% "
                 "signal efficiency (__WPHQQL__\\% for $H\\to\\ell\\nu qq'$, __WPTBL__\\% for $t\\to b\\ell\\nu$) on "
                 "$D_S = g_S/(g_S+g_{\\mathrm{QCD}})$ of equation~(\\ref{eq:discr}). Published rows: trained from scratch on 2 million jets and scored on "
                 "ParT's \\LitPartJcTestMillion-million-jet test set (ParT on 100 million jets reaches __PART100__); "
                 "ours fine-tuned on one million jets and scored on two million (the AUC on __AUCJETS__ of them). "
                 + SPREAD3 + " From scratch: three seeds. Bold: the best value of each column, and every value within "
                 "one standard deviation of it. $^\\dagger$: fewer than $\\nmin$ QCD jets pass in at least "
                 "one run (equation~(\\ref{eq:cut})), so that run's rejection has a counting uncertainty above __MINBKGPC__\\%; these columns "
                 "carry no claim."),
    "jetclass_all": ("tab:jetclassall", "JetClass fine-tuning as table~\\ref{tab:jetclass}, every configuration. "
                     + SPREAD + " Bold and $^\\dagger$ as there."),
    "jetclass_probe": ("tab:jetclassprobe", "JetClass linear probes on the frozen features, largest training set, in "
                       "the columns of table~\\ref{tab:jetclass} (AUC one against one where the fit records it). "
                       + SPREAD + " $^\\dagger$ as there."),
    "jetclass_nbkg": ("tab:jetclassnbkg", "QCD jets passing each signal class's working point after fine-tuning, the "
                      "fewest over the runs of each configuration: the counts behind the rejections of "
                      "table~\\ref{tab:jetclass}."),
    "per_class": ("tab:perclass", "Linear probes per class of JetClass-II (class-balanced test set, largest training "
                  "set): the AUC of the class against the rest for the \\Kfull-class model (mean $\\pm$ standard "
                  "deviation over its runs) and the ratio of $1-\\mathrm{AUC}$ to it of every configuration, run by "
                  "run and averaged; p1--p5 are the random partitions, then the partitions with no, a $b$-quark and a "
                  "random flavor boundary. Classes grouped by their \\Kcoarse-class group."),
    "per_run": ("tab:perrun", "Every pretraining run's values at the largest training set: linear probe and "
                "fine-tuning. Linear probes on top and quark/gluon with the missing inputs substituted as chosen on "
                "validation (\\LpSubstitutionTop{} for top, \\LpSubstitutionQg{} for quark/gluon); fine-tuning with "
                "them set to zero."),
    "rand_prong": ("tab:randprong", "The random partitions on JetClass: which prong-count boundaries each keeps "
                   "($\\bullet$: no group holds classes of both prong counts), the groups that mix prong counts, and "
                   "its fine-tuned rejection of QCD per signal class over that of the \\Kfull-class run of its seed. "
                   "$^\\dagger$: fewer than $\\nmin$ QCD jets pass in one of the two runs (equation~(\\ref{eq:cut}))."),
    "lrsweep": ("tab:lrsweep", "Learning-rate choice at $10^{4}$ training jets: the best validation accuracy of each "
                "rate on top and quark/gluon and their sum. $^{\\ast}$: the rate chosen, the largest sum; the test "
                "jets play no part."),
    "inputs": ("tab:inputs", "The per-particle inputs: which the top and the quark/gluon data provide ($\\bullet$), "
               "where a zero lies after the network's input normalization (in standard deviations of the training "
               "particles), and what the mean and random-draw substitutions put in a missing slot. A random draw is "
               "averaged over four draws."),
    "partrepro": ("tab:partrepro", "ParT's top-tagging training script on our cluster, then with Sophon's output head, "
                  "then also with Sophon's rescaling of every jet's four-momenta to $p_{\\mathrm{T}}=500$~GeV "
                  "(cumulative, one run each), beside ParT's published row and our network trained from scratch, "
                  "which uses the rescaling."),
    "xbx": ("tab:xbx", "Resonance tagging in the soft-drop mass window, \\XbxWindowLo--\\XbxWindowHi~GeV: signal "
            "efficiency (\\%) at the QCD efficiency given, mean $\\pm$ standard deviation over the pretraining runs. "
            "Transfer: a classifier on the frozen features, its seeds averaged per run (the randomly initialized "
            "network: the linear probe). Direct: Sophon's discriminant from the output scores, where $X\\to b\\bar b$ "
            "is a class of its own."),
    "xbx_rej": ("tab:xbxrej", "Resonance tagging in the mass window: rejection $1/\\epsB$ of QCD jets at 50\\% and "
                "30\\% signal efficiency. Transfer: the classifier on frozen features; direct: Sophon's discriminant "
                "from the output scores; closest node: the pretrained $X\\to bq$ output against QCD. " + SPREAD),
    "compute": ("tab:compute", "GPU types and GPU-hours of every pretraining run, summed over the training and "
                "validation passes of its epochs as each epoch recorded its device, and the GPU types its "
                "fine-tuning ran on."),
}
# (caption key, table file, wide, caption fill) of the tables written as placeholders until their data exist
PLACEHOLDERS = (("probes", "mlst_probes", True, None), ("probes_n", "mlst_probes_n", True, None),
                ("pairs", "mlst_pairs", True, {"__ZERO__": ""}), ("mlp", "mlst_mlp", True, None), ("bn", "mlst_bn", False, None),
                ("balance", "mlst_balance", False, None), ("fill", "mlst_fill", True, None),
                ("mass", "mlst_mass", False, None), ("mass_detail", "mlst_mass_detail", True, None),
                ("sculpt", "mlst_sculpt", False, {"__SCULPT_BINS__": r"\pending{bins}"}), ("ft", "mlst_ft", True, None),
                ("ft_rej", "mlst_ft_rej", True, {"__FEWEST__": r"\pending{n}"}), ("boot", "mlst_boot", True, {"__JC2COMP__": "class-balanced"}),
                ("bench_top", "mlst_bench_top", True, None), ("bench_qg", "mlst_bench_qg", True, None),
                ("xbx", "mlst_xbx", False, None), ("xbx_rej", "mlst_xbx_rej", True, None),
                ("rand_prong", "mlst_rand_prong", True, None),
                ("lrsweep", "mlst_lrsweep", False, None), ("inputs", "mlst_inputs", True, None),
                ("partrepro", "mlst_partrepro", False, None), ("compute", "compute", False, None),
                ("jetclass_all", "mlst_jetclass_all", True, None), ("jetclass_probe", "mlst_jetclass_probe", True, None),
                ("jetclass_nbkg", "mlst_jetclass_nbkg", True, None),
                ("jetclass", "mlst_jetclass", True,
                 {k: r"\pending{n}" for k in ("__WPHQQL__", "__WPTBL__", "__MINBKG__", "__MINBKGPC__", "__PART100__",
                                              "__AUCJETS__")}))
LONGTABLES = (("per_class", "mlst_per_class"), ("per_run", "mlst_per_run"))

# The macros the new paper's outline names (docs/PAPER_OUTLINE.md, PI decisions 2026-11-02), with what each
# waits on: one still not emitted after build() becomes a red marker and is listed as missing, so a
# sentence written with it compiles and shows what it waits for. (name, waits on, twins)
REQUIRED = (
    ("DesignNTrainSelected", "a committed count of the selected pretraining jets", ""),
    ("FtBudgetEvalSmall", "build_ft_budget_jobs.py", ""),
    ("LpSubstitutionTop", "fill fits", ""), ("LpSubstitutionQg", "fill fits", ""),
    ("MlstFtJcOneCoarseNRejDrop", "JetClass curves", ""), ("MlstFtJcOneCoarseRejDropClasses", "JetClass curves", ""),
    ("MlstFtJcOneRejHfourqCoarseOverFull", "JetClass curves", "Pr"), ("MlstFtJcOneRejHfourqMidOverFull", "JetClass curves", "Pr"),
    ("MlstFtJcOneRejTbqqRandOverFull", "JetClass curves", "Pr"), ("MlstFtJcOneAccRandMinusCoarse", "JetClass curves", ""),
    ("MlstFtCmpJcTwoCoarseOverFullESix", "class-balanced JetClass-II fine-tuning", "Pr"),
    ("MlstPerClassRandFlavMergedOverApart", "probes", "Pr"), ("MlstPerClassCoarseNDoubled", "probes", ""),
    ("MlstLpBbQgZeroAccDropFull", "pair probes with the quark/gluon inputs zeroed (open item O10, not run)", ""),
    ("MlstLpBbQgZeroAccDropCoarse", "pair probes with the quark/gluon inputs zeroed (open item O10, not run)", ""),
    ("MlstPairRandMergedOverApartBb", "probes", "Pr"),
    ("MlstFtTopRthirtyFullOverScratch", "fine-tuning read-outs", "Pr"), ("MlstFtTopRthirtyScratch", "fine-tuning read-outs", ""),
    ("MlstFtTopRthirtyCoarseOverFull", "fine-tuning read-outs", "Pr"),
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
    ("MlstCmpJcTwoResMassOverRes", "probes", "Pr PrMax"),
    ("MlstSculptFullOverFloorOne", "sculpting", "Pr"), ("MlstSculptCoarseOverFullOne", "sculpting", "Pr"),
    ("MlstSculptMidOne", "sculpting", ""), ("MlstSculptCoarseMassOverCoarseOne", "sculpting", "Pr PrMax"),
    ("MlstSculptDiffEffLo", "sculpting curves", ""), ("MlstSculptDiffEffHi", "sculpting curves", ""),
    ("MlstSculptLowestEff", "sculpting curves", ""), ("MlstSculptLowestNPass", "sculpting curves", ""), ("MlstSculptLowestMaxFloorSd", "sculpting curves", ""),
    ("XbsRthirtyFullOverScratch", "X->bs fits and from-scratch runs", "Pr"),
    ("XbsRthirtyCoarseOverFull", "X->bs fits", "Pr"), ("XbsRthirtyNodeOverTransferFull", "X->bs fits", "Pr"),
    ("XbbRthirtyDirectOverTransferFull", "X->bb fits", "Pr"), ("XbbRthirtyTransferCoarseOverFull", "X->bb fits", "Pr"),
    ("MlstComputeTotalGpuHours", "compute records", ""), ("MlstGpuTypeBound", "probes", ""),
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
    *((f"MlstLpEemmAuc{w}", "the ee/mumu probe (fits_v4_sib)", "") for w in ("Full", "Mid", "Coarse")),
    *((f"MlstPairEemm{k}", "the ee/mumu probe (fits_v4_sib)", "") for k in ("NTrain", "NVal", "NTestPerSide")),
)


def _float(key: str, body: str, wide: bool = False, fill: dict | None = None, notes: list | None = None) -> str:
    env = "table*" if wide else "table"
    label, cap = CAPTIONS[key]
    for k, v in (fill or {}).items():
        cap = cap.replace(k, v)
    assert "__" not in cap, cap
    boxes = [r"\begin{adjustbox}{max width=\linewidth}" + "\n" + b.strip() + "\n" + r"\end{adjustbox}"
             for b in body.split("%SPLIT%")]
    tail = [rf"\par\smallskip{{\footnotesize {n}}}" for n in notes or []]
    return "\n".join([rf"\begin{{{env}}}[t]", r"\centering", r"\small", rf"\caption{{{cap}}}",
                      rf"\label{{{label}}}", "\n\\par\\medskip\n".join(boxes), *tail, rf"\end{{{env}}}", ""])


def _longtable(key: str, body: str) -> str:
    return body.replace("__CAPTION__", CAPTIONS[key][1])


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
    out = {"tables/mlst_probes.tex": _float("probes", table_probes(fits, runs, fmt_pm, rows_init, full_count, fill,
                                                                   chosen), True),
           "tables/mlst_probes_n.tex": _float("probes_n", table_probes_n(fits, runs, fmt_pm, MAIN + MASS + (INIT, INIT_BN),
                                                                         fill, chosen), True),
           "tables/mlst_pairs.tex": _float("pairs", table_pairs(fits, runs, fmt_pm, status, MAIN + MASS + RAND + FLAV), True,
                                           fill={"__ZERO__": zero_note(fits, runs)}),
           "tables/mlst_balance.tex": _float("balance", table_balance(fits3, fits_v4, runs, fmt_pm, ALL_ROWS + (INIT_BN,)))}
    comparisons(em, fits, runs, status, index, fmt_pm)
    eemm_macros(em, fits, runs, index, fmt_pm, fmt_int)
    probe_abs_macros(em, fits, runs, index, fmt_pm)
    if fits3:
        bn_macros(em, fits3, runs, index)
        out["tables/mlst_bn.tex"] = _float("bn", table_bn(fits3, runs, fmt_pm, ALL_ROWS))
    V4.gpu_bound_macro(em, fits, runs, comp, index)
    per_class_macros(em, fits, runs, root, index, fmt_pm, native_tex)
    out["tables/mlst_per_class.tex"] = _longtable("per_class", table_per_class(fits, runs, fmt_pm, root, native_tex))
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
    trained = {(m, ds, n) for (m, ds, n) in UNCONVERGED_SEEN if not m.startswith(("init-", "initbn-"))}
    em.macro("MlstProbeUnconverged", str(len(trained)), index, "linear probes of pretrained models that did not converge (count)")
    em.macro("MlstProbeUnconvergedInit", str(len(UNCONVERGED_SEEN) - len(trained)), index,
             "linear probes of the randomly initialized networks that did not converge (count)")
    cells = [c for m, d in fits.items() for c in _fit_cells(d["cells"].get(_ck(m), {}))]
    edge = [c for m, d in fits.items() for c in _fit_cells(d["cells"].get(_ck(m), {})) if c["C"] == max(d["c_grid"])]
    em.macro("MlstProbeFits", str(len(cells)), index, "selected linear-probe fits of the reported checkpoints (count)")
    em.macro("MlstProbeEdgeC", str(len(edge)), index, "of these, fits whose selected C is the largest of the grid (count)")
    mlp = load_mlp(data)
    if mlp:
        out["tables/mlst_mlp.tex"] = _float("mlp", table_mlp(mlp, runs, fmt_pm, rows_init, raw_v4), True)
        mlp_macros(em, mlp, runs, index, fmt_pm)
    else:
        missing.append(f"nonlinear probes -- experiments/FIGS/data/v2/{MLP_FITS_V4}/")
    out["tables/mlst_fill.tex"] = _float("fill", table_fill(raw_v4, fill, fits, runs, fmt_pm, ALL_ROWS + (INIT_BN,)), True)
    fill_macros(em, raw_v4, fill, fits, runs, index, fmt_pm, missing)
    loss_share_macros(em, comp, runs, index, fmt_pm, root)
    atlas = _load(data / "head" / "atlas_res")
    if atlas:
        out["tables/mlst_mass.tex"] = _float("mass", table_mass(atlas, runs, fmt_pm))
        out["tables/mlst_mass_detail.tex"] = _float("mass_detail", table_mass_detail(atlas, runs, fmt_pm), True)
        mass_macros(em, atlas, runs, index, fmt_pm)
    else:
        missing.append("ATLAS mass resolution -- experiments/FIGS/data/v2/head/atlas_res/")
    head = _load(data / "head")
    sculpt_v2 = {k.removesuffix("_sculpting_v2"): v for k, v in _load(data / "head" / "curves_v2").items()
                 if k.endswith("_sculpting_v2")}
    sculpt = {**{k.removesuffix("_sculpting"): v for k, v in head.items() if k.endswith("_sculpting")}, **sculpt_v2}
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
        sculpt_macros(em, sculpt, sculpt_v2, runs, index, fmt_pm)
    else:
        missing.append("mass sculpting -- experiments/FIGS/data/v2/head/*_sculpting.json")
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
        out["tables/mlst_ft.tex"] = _float("ft", table_ft(per, runs, fmt_pm, ALL_ROWS), True)
        body, fewest = table_ft_rej(per, runs, fmt_pm, ALL_ROWS)
        out["tables/mlst_ft_rej.tex"] = _float("ft_rej", body, True,
                                               fill={"__FEWEST__": fmt_int(fewest) if fewest else "--"})
        ft_macros(em, per, runs, index, fmt_pm)
        topcheck_macros(em, data, per, index, fmt_pm, missing)
        out["tables/mlst_bench_top.tex"] = _float("bench_top", V4.table_bench(
            "top", per, runs, MAIN, LABEL, SCRATCH, fmt_pm, _two, kin=(LABEL["KIN"] + " (diagnostic)", KIN_INIT),
            zero=r"\FillTopNZero{}"), True, notes=[partrepro_note()])
        out["tables/mlst_bench_qg.tex"] = _float("bench_qg", V4.table_bench(
            "qg", per, runs, MAIN, LABEL, SCRATCH, fmt_pm, _two, herwig=True, zero=r"\FillQgNZero{}"), True)
    gap_macros(em, fits, per if tag else {}, runs, index, fmt_pm)
    out["tables/mlst_per_run.tex"] = _longtable("per_run", table_per_run(fits, per if tag else {}, runs, fmt_pm, fill,
                                                                         chosen))
    xbx_section(em, root, data, index, runs, missing, out, fmt_pm, fmt_int)
    t = partrepro(em, data, per if tag else {}, fmt_pm, missing)
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
    full = [data / "curves" / f"full_{t}.json" for t in ("best70_leg2", "ref_v100_jc1_leg2")]
    if all(f.exists() for f in full):
        pre, ref = (json.loads(f.read_text())["cells"] for f in full)
        pooled = {**runs, "RAND": [m for a in RAND for m in runs[a]]}
        lab = {**LABEL, "RAND": "random partitions (all five)"}
        blk = next((b for c in pre.values() for b in [c.get(JC1_N, {}).get("s1", {}).get("jc1")] if b), None)
        lit = next(r for r in json.loads(V4.LIT.read_text())["jetclass"] if r["method"] == "ParT")
        jfill = {"__WPHQQL__": f"{100 * V4.LPV_EPS['label_Hqql']:g}", "__WPTBL__": f"{100 * V4.LPV_EPS['label_Tbl']:g}",
                 "__MINBKG__": str(V4.JC1_MIN_BKG), "__MINBKGPC__": f"{100 / math.sqrt(V4.JC1_MIN_BKG):.0f}",
                 "__PART100__": f"an accuracy of {lit['acc']} and an AUC of {lit['auc_ovo']}",
                 "__AUCJETS__": fmt_int(blk["n_jets_auc"]) if blk else r"\pending{n}"}
        out["tables/mlst_jetclass.tex"] = _float("jetclass", V4.table_jetclass(
            pre, ref, pooled, MAIN, lab, SCRATCH, 1_000_000, fmt_pm, _two), True, fill=jfill)
        out["tables/mlst_jetclass_all.tex"] = _float("jetclass_all", V4.table_jetclass(
            pre, ref, pooled, MAIN + MASS + ("RAND",) + FLAV, lab, SCRATCH, 1_000_000, fmt_pm, _two, literature=False), True)
        out["tables/mlst_jetclass_nbkg.tex"] = _float("jetclass_nbkg", V4.table_jetclass_nbkg(
            pre, ref, pooled, MAIN + MASS + ("RAND",) + FLAV, lab, SCRATCH, 1_000_000), True)
        jetclass_macros(em, pre, runs, full[0], fmt_pm, words)
        out["tables/mlst_rand_prong.tex"] = _float("rand_prong", table_rand_prong(pre, runs, root, fmt_pm), True)
    else:
        missing.append("JetClass per-class curves -- experiments/FIGS/data/v2/curves/full_*.json")
    out["tables/mlst_jetclass_probe.tex"] = _float("jetclass_probe", table_jetclass_probe(fits, runs, fmt_pm, rows_init), True)
    if comp:
        ftg, nodev = ft_gpu_types(data)
        if nodev:
            missing.append(f"GPU of {sum(nodev.values())} fixed-budget fine-tuning cells (JetClass, JetClass-II): "
                           "experiments/FT/budget_metrics.py records it for top and quark/gluon only")
        body, hours = table_compute(comp, runs, fmt_int, extra=((LABEL["KIN"], KIN_MODEL),), ft_gpus=ftg,
                                    ft_nodev=nodev)
        out["tables/compute.tex"] = _float("compute", body, True)
        compute_macros(em, comp, runs, hours, index, fmt_int)
        if len(set(pre_runs) & set(comp)) < len(pre_runs):
            missing.append(f"compute records: {len(set(pre_runs) & set(comp))} of {len(pre_runs)} paper runs")
    required_pass(em, index, missing)
    # a table whose data is not pulled yet is a marked placeholder, never the file an older generator left behind
    for key, name, wide, fill_ in PLACEHOLDERS:
        out.setdefault(f"tables/{name}.tex", _float(key, r"\pending{no result yet}", wide, fill_))
    for key, name in LONGTABLES:
        out.setdefault(f"tables/{name}.tex", _float(key, r"\pending{no result yet}", True))
    return out
