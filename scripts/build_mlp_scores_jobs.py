#!/usr/bin/env python3
"""The MLP probes of the three linear-probe nulls the main text states (PI 2026-11-23, LEDGER DL-9; rule CS-18), with
their per-jet test scores (experiments/EVAL/mlp_scores.py), and their paired test-jet bootstraps under SC-1
(experiments/STATS/paired_bootstrap.py). CPU only; every fit of every family on one CPU model, the MLP family's (PF-1,
build_refit_v5_jobs.FAMILY_CPU["mlp"]: the EPYC 7402P), the pod refusing any other; tab:mlp's protocol (three seeds,
four threads, the plateau rule); the same runs as the linear-probe rows. New folders only; a fit or file that exists
is skipped, so a retried pod does only what is missing.

  bs3-mlp-<family>-raunav   Indexed, one a family, ROWS: jc2bal (188 and 162 classes: accuracy, the null of results_jc), coarse (188, 162,
                            43 and 17 classes on the 17 merged groups: tab:coarse's nulls), bench (the 27 networks of
                            Fig. 2 on top and quark/gluon, Pythia and Herwig: the R30 nulls of its caption, under the
                            linear probes' substitution of the missing inputs)
  bs3-mlp-smoke-raunav, bs3-mlp-smoke-bench-raunav   one row of the same body (SMOKES), its peak memory and
                            seconds, to size the family's job, which then leaves that row out (--skip-smoke), so no
                            fit runs in two jobs
  bs3-boot-jc2-raunav, bs3-boot-bench-raunav   one pod each once its fits exist: BOOT_OUT's files (BOOT_PARTS)
  bs3-analytic-raunav       one pod: SC-2 #1, the transfer-head term against the paired bootstrap (ANALYTIC_OUT)

    python3 scripts/build_mlp_scores_jobs.py --tag mtx-sN.NN --what fit --family coarse --mem 6Gi [--parallel 8]
        [--skip-smoke]      (--what smoke | smoke-bench | fit | boot | analytic)
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "experiments" / "STATS" / "k8s"


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


R5 = _load("build_refit_v5_jobs", "scripts/build_refit_v5_jobs.py")
BT = _load("build_boot_tests_jobs", "scripts/build_boot_tests_jobs.py")
CPU_MODEL = R5.FAMILY_CPU["mlp"]
THREADS = "4"                            # probe.MLP_THREADS: the pod's CPU request matches what the fit uses
LINE = 88                                # the brief's stop line for these writes (below the 90% finishing line)
OUT = "/data/results/eval/bs3_mlp"       # <OUT>/<family>/<model>.json, <OUT>/<family>/probs/<model>/best70/*.npz
BOOT = "/data/results/bootstrap/l3_bs3"
B = BT.B
ARMS = R5.BOOTALL_ARMS
# jc2bal: 188 and 162 classes (DL-9, the accuracy null) and the mass networks with their label sets (DL-3b: the MLP
# AUC rows 162 + mass - 162 and 17 + mass - 17 of tab:tests)
# bench: the fill chosen on validation (CS-8b, PI 2026-11-24: one configuration for Table 10 and the DL-9 rows), with
# the random networks; bcpair: SC-2c's X->bc vs bq, cs pair MLP of 17 and 188 classes
FAMILY_ARMS = {"jc2bal": ("l188", "l162", "l162mass", "r16q1", "r16q1mass"), "coarse": ("l188", "l162", "r42q1", "r16q1"),
               "bench": tuple(ARMS) + ("init", "initbn"), "bcpair": ("l188", "r16q1"), "eepair": ("l188", "r16q1"),
               "bench0": tuple(ARMS) + ("init", "initbn"), "kin": ("kin",)}
INIT = {"init": "init", "initbn": "init_bn"}     # the random networks' checkpoints (init-sK's two)


def row_of(arm: str, r: str) -> tuple[str, str]:
    """(model, checkpoint) of a run of an arm."""
    return (f"init-{r.rsplit('-', 1)[1]}", INIT[arm]) if arm in INIT else (f"mtx-{r}", "best70")


ARMS = {**ARMS, "init": ("init-s1", "init-s2", "init-s3"), "initbn": ("initbn-s1", "initbn-s2", "initbn-s3"),
        "kin": ("l188kin-s1",)}
ROWS = [(f, *row_of(a, r)) for f, arms in FAMILY_ARMS.items() for a in arms for r in ARMS[a]]
# smokes: jc2bal, the largest output and a bit-for-bit check against fits_v5_mlp; bench, the largest training set
SMOKES = {"jc2bal": ("jc2bal", "mtx-l188-s1", "best70"), "bench": ("bench", "mtx-l188-s1", "best70")}
SMOKE_NAME = {"jc2bal": "bs3-mlp-smoke-raunav", "bench": "bs3-mlp-smoke-bench-raunav"}
FILES = {"jc2bal": ("jc2bal_N1000000",), "coarse": ("jc2bal_R16_Q1_N1000000",),
         "bench": ("top_N1200000", "qg_N1600000", "qg_herwig_N1600000"),
         "bench0": ("top_N1200000", "qg_N1600000", "qg_herwig_N1600000"), "kin": ("top_N1200000",),
         "bcpair": ("jc2bal_pairs_bc_vs_bq_cs",), "eepair": ("jc2bal_pairs_ee_vs_mm",)}
# the bootstraps: output -> (family, npz name, the differences (first arm, second arm), flags)
BENCH_DIFFS = tuple((a, "l188") for a in FAMILY_ARMS["bench"][:10] if a != "l188")
# every pair of Table 10's rows, 188 classes always the second side (Delta = the other row - 188 classes)
PAIRS0 = tuple((b, a) if a == "l188" else (a, b) for a, b in itertools.combinations(FAMILY_ARMS["bench0"], 2))
PAIRSF = tuple((b, a) if a == "l188" else (a, b) for a, b in itertools.combinations(FAMILY_ARMS["bench"], 2))
MASS_OUT = f"{BOOT}/mlp_jc2bal_N1000000.mass.json"
MASS_CACHE = "/data/results/bootstrap_cache/bs3_mlp_jc2bal_N1000000"     # outside bootstrap/: never pulled
BOOT_OUT = {f"{BOOT}/mlp_jc2bal_N1000000.json": ("jc2bal", "jc2bal_N1000000", (("l162", "l188"),), "--headline-only"),
            f"{BOOT}/mlp_jc2bal_R16_Q1_N1000000.json": ("coarse", "jc2bal_R16_Q1_N1000000",
                                                        (("l162", "l188"), ("r42q1", "l188"), ("r16q1", "l188")),
                                                        "--headline-only"),
            **{f"{BOOT}/mlp_{n}.json": ("bench", n, BENCH_DIFFS, "--headline") for n in FILES["bench"]},
            # DL-3b: the mass networks against their label sets, every metric tab:mlp prints (AUC, resonant, QCD)
            MASS_OUT: ("jc2bal", "jc2bal_N1000000", (("l162mass", "l162"), ("r16q1mass", "r16q1")),
                       "--headline --native-groups"),
            # CS-8b: every pair of Table 10's rows on the fill chosen on validation (the configuration Table 10 takes)
            **{f"{BOOT}/mlpf_{n}.json": ("bench", n, PAIRSF, "--headline") for n in FILES["bench"]},
            # SC-2c: the X->bc vs bq, cs pair MLP, 17 - 188 classes
            f"{BOOT}/mlp_jc2bal_pairs_bc_vs_bq_cs.json": ("bcpair", "jc2bal_pairs_bc_vs_bq_cs", (("r16q1", "l188"),),
                                                          "--headline"),
            # fifth review B2: the X->ee vs X->mumu pair MLP beside the 17-class null
            f"{BOOT}/mlp_jc2bal_pairs_ee_vs_mm.json": ("eepair", "jc2bal_pairs_ee_vs_mm", (("r16q1", "l188"),), "--headline"),
            # CS-5b: every pair of Table 10's rows on its printed zero-fill fits
            **{f"{BOOT}/mlp0_{n}.json": ("bench0", n, PAIRS0, "--headline") for n in FILES["bench0"]}}


def models(arm: str) -> list[str]:
    """The names a bootstrap gives an arm's runs: mtx-<run>, init-sK, initbn-sK (paired_bootstrap's)."""
    return [r if arm in INIT else f"mtx-{r}" for r in ARMS[arm]]


# bench: the linear probes' substitution of the missing inputs (mlp_scores.FILL; third review of 5f4e1b8, B3(c)), after
# the checkpoint (best70, or the random networks' init and init_bn)
FILL_OF = {"top_N1200000": "itop4", "qg_N1600000": "nqg", "qg_herwig_N1600000": "nqg"}


def npz(family: str, model: str, name: str) -> str:
    """A model's scores as mlp_scores writes them; model as models() names it (initbn-sK: init-sK at init_bn)."""
    run, ck = (model.replace("initbn-", "init-"), "init_bn") if model.startswith("initbn-") else (
        (model, "init") if model.startswith("init-") else (model, "best70"))
    return f"{OUT}/{family}/probs/{run}/{f'{ck}.{FILL_OF[name]}' if family == 'bench' else ck}/{name}.npz"


def guard() -> str:
    return ('USED=$(df --output=pcent /data | tail -1 | tr -dc 0-9)\n'
            f'[ "${{USED}}" -le {LINE} ] || {{ echo "FATAL: /data at ${{USED}}%"; exit 42; }}')


def fit_row(family: str, model: str, ckpt: str = "best70") -> str:
    f = f"{OUT}/{family}/{model}.json" if ckpt == "best70" else f"{OUT}/{family}/{model}.{ckpt}.json"
    return (f'[ -f {f} ] && {{ echo "{model} {ckpt} done"; exit 0; }}; ' + BT.peak(
        f"python3 experiments/EVAL/mlp_scores.py --family {family} --model {model} --checkpoint {ckpt} "
        f"--out {OUT}/{family}"))


def _spec(tag: str, name: str, rows: list[str], parallelism: int, mem: str, cpu: str, what: str) -> str:
    body = f"""set -euo pipefail
{R5.JB._start(tag)}
{R5.threads(cpu)}
{guard()}
{R5.case_block(rows)}echo "BS3 ${{IDX}} COMPLETE"
"""
    text = R5.indexed(name, body, len(rows), parallelism, mem, cpu, what).replace(
        "# GENERATED by scripts/build_refit_v5_jobs.py", "# GENERATED by scripts/build_mlp_scores_jobs.py")
    return R5.pin_cpu(text, CPU_MODEL)


def fit_rows(family: str, skip_smoke: bool, arms=None) -> list[tuple[str, str]]:
    """The family's rows (arms: only those arms' runs); with skip_smoke, without the row a smoke job fits (SMOKES)."""
    keep = {row_of(a, r) for a in arms for r in ARMS[a]} if arms else None
    return [r for r in ROWS if r[0] == family and not (skip_smoke and r in SMOKES.values())
            and (keep is None or tuple(r[1:]) in keep)]


def fit_spec(tag: str, family: str, mem: str, parallelism: int, skip_smoke: bool, arms=None, suffix: str = "") -> str:
    """bs3-mlp-<family><suffix>-raunav: one job a family (or a later set of its arms), so each family's pods are
    sized to its own measured peak."""
    rows = [fit_row(*r) for r in fit_rows(family, skip_smoke, arms)]
    return _spec(tag, f"bs3-mlp-{family}{suffix}-raunav", rows, parallelism, mem, THREADS,
                 f"MLP probes with per-jet scores ({family}) for the main text's linear-probe nulls (PI 2026-11-23, DL-9)")


def smoke_spec(tag: str, mem: str, which: str = "jc2bal") -> str:
    return _spec(tag, SMOKE_NAME[which], [fit_row(*SMOKES[which])], 1, mem, THREADS,
                 "MLP probes with per-jet scores, smoke: one fit, its peak memory and seconds")


# the bootstrap jobs: JetClass-II (jc2bal and coarse) and bench, each launched once its fits exist
BOOT_PARTS = {"jc2": (f"{BOOT}/mlp_jc2bal_N1000000.json", f"{BOOT}/mlp_jc2bal_R16_Q1_N1000000.json"),
              "bench": tuple(f"{BOOT}/mlp_{n}.json" for n in FILES["bench"]),
              "bench0": tuple(f"{BOOT}/mlp0_{n}.json" for n in FILES["bench0"]),
              "benchf": tuple(f"{BOOT}/mlpf_{n}.json" for n in FILES["bench"]),
              "bcpair": (f"{BOOT}/mlp_jc2bal_pairs_bc_vs_bq_cs.json",),
              "eepair": (f"{BOOT}/mlp_jc2bal_pairs_ee_vs_mm.json",),
              "jc2mass": (MASS_OUT,)}


def boot_outs(part: str) -> dict:
    return {o: BOOT_OUT[o] for o in BOOT_PARTS[part]}


def boot_inputs(part: str | None = None) -> list[str]:
    outs = BOOT_OUT if part is None else boot_outs(part)
    return [npz(fam, m, n) for fam, n, d, _ in outs.values() for a in {x for p in d for x in p} for m in models(a)]


def boot_spec(tag: str, mem: str, part: str) -> str:
    """bs3-boot-<part>-raunav: one pod, the part's BOOT_OUT files once every input exists."""
    lines = [BT.check_inputs(boot_inputs(part)), f"mkdir -p {BOOT}"]
    for out, (fam, n, d, flags) in boot_outs(part).items():
        arms = list(dict.fromkeys(x for p in d for x in p))
        ms = " ".join(f"--model {m}={npz(fam, m, n)}" for a in arms for m in models(a))
        lines.append(f"[ -e {out} ] || " + BT.peak(f"python3 experiments/STATS/paired_bootstrap.py --task npz {ms} "
                                                   f"{flags} {BT.diffs(d)} --b {B} --out {out}"))
    return _spec(tag, f"bs3-boot-{part}-raunav", ["\n  ".join(lines)], 1, mem, "1",
                 f"paired test-jet bootstraps of the MLP probes' differences ({part}; SC-1, B = 200)")


def boot_mass_spec(tag: str, mem: str, parts: int = 3) -> str:
    """bs3-boot-jc2mass-raunav: the macro AUC over 188 classes takes minutes per network and resample set, so indices
    0..parts-1 fill MASS_CACHE (paired_bootstrap --cache-only, a share of the networks each) and the last index writes
    MASS_OUT from the cache once every network is in it."""
    fam, n, d, flags = BOOT_OUT[MASS_OUT]
    ms = [m for a in dict.fromkeys(x for p in d for x in p) for m in models(a)]
    inp = {m: npz(fam, m, n) for m in ms}
    pb = "python3 experiments/STATS/paired_bootstrap.py --task npz"
    rows = [" && ".join([BT.check_inputs([inp[m] for m in ms[i::parts]]),
                         BT.peak(f"{pb} " + " ".join(f"--model {m}={inp[m]}" for m in ms[i::parts])
                                 + f" {flags} --b {B} --cache {MASS_CACHE} --cache-only")]) for i in range(parts)]
    wait = (f'T0=$(date +%s); while [ $(for m in {" ".join(ms)}; do [ -e {MASS_CACHE}/${{m}}.npz ] || echo x; done | wc -l) -gt 0 ]; '
            'do [ $(( $(date +%s) - T0 )) -lt 36000 ] || { echo "FATAL: cache incomplete"; exit 42; }; sleep 120; done')
    rows.append("\n  ".join([f"mkdir -p {BOOT}", wait, f"[ -e {MASS_OUT} ] || " + BT.peak(
        f"{pb} " + " ".join(f"--model {m}={inp[m]}" for m in ms) + f" {flags} {BT.diffs(d)} --b {B} "
        f"--cache {MASS_CACHE} --out {MASS_OUT}")]))
    return _spec(tag, "bs3-boot-jc2mass-raunav", rows, parts, mem, "1",
                 "DL-3b: paired bootstrap of the MLP AUC, the mass networks against their label sets")


def boot_mass_helper_spec(tag: str, mem: str, skip: int = 3) -> str:
    """bs3-boot-jc2mass-helper-raunav: one pod a network for the networks after the first skip (the cache pods' later
    networks), into the same MASS_CACHE; a network already cached is skipped, so the cache pods then find them there."""
    fam, n, d, flags = BOOT_OUT[MASS_OUT]
    ms = [m for a in dict.fromkeys(x for p in d for x in p) for m in models(a)][skip:]
    pb = "python3 experiments/STATS/paired_bootstrap.py --task npz"
    rows = [" && ".join([BT.check_inputs([npz(fam, m, n)]), f"[ -e {MASS_CACHE}/{m}.npz ] || " + BT.peak(
        f"{pb} --model {m}={npz(fam, m, n)} {flags} --b {B} --cache {MASS_CACHE} --cache-only")]) for m in ms]
    return _spec(tag, "bs3-boot-jc2mass-helper-raunav", rows, len(rows), mem, "1",
                 "DL-3b: the mass bootstrap's later networks, one pod each, into the same cache")

# ------------------------------------------------------------------ fourth review B5: 188 with draws against kinematic inputs
KIN_OUT = f"{BOOT}/mlp_top_N1200000.kin.json"


def boot_kin_spec(tag: str, mem: str) -> str:
    """bs3-boot-kin-raunav: the MLP of 188 classes with random draws (bench) against the kinematic-inputs network's
    (kin, one run: the sigma lacks its run spread, a lower bound), on the same top test jets, as the linear-probe row
    bootstrap/l3_v5/probe_top_N1200000.kin.json 188draws-kin."""
    ins = {**{m: npz("bench", m, "top_N1200000") for m in models("l188")}, "mtx-l188kin-s1": npz("kin", "mtx-l188kin-s1", "top_N1200000")}
    cmd = ("python3 experiments/STATS/paired_bootstrap.py --task npz " + " ".join(f"--model {m}={p}" for m, p in ins.items())
           + f" --headline --diff 188draws-kin={','.join(models('l188'))}/mtx-l188kin-s1 --b {B} --out {KIN_OUT}")
    body = "\n  ".join([BT.check_inputs(ins.values()), f"mkdir -p {BOOT}", f"[ -e {KIN_OUT} ] || " + BT.peak(cmd)])
    return _spec(tag, "bs3-boot-kin-raunav", [body], 1, mem, "1",
                 "fourth review B5: the MLP, 188 classes with random draws against the kinematic-inputs network (top R30)")

# ------------------------------------------------------------------ CS-15b: fine-tuned against linear probe (after the reruns)
CS15B_OUT = f"{BOOT}/ft_lp_jc2bal_N1000000.json"
CS15B_ARMS = tuple(a for a in ARMS if a not in INIT and a != "kin")


def cs15b_inputs() -> dict[str, tuple[str, str]]:
    """{arm: [(ft name, cell), (lp name, npz)]}: each pretrained network's JetClass-II 10^6 fine-tuning cell at its row's
    recorded rate (build_ft_budget_jobs.paper_fixed_cells, read only) and its linear probe on the same 903,922 test jets
    (fits_v5_probs); a network without its cell is an error here, not a silent gap."""
    fb = _load("build_ft_budget_jobs", "scripts/build_ft_budget_jobs.py")
    cells = {c["init"]: c["out"] for c in fb.paper_fixed_cells() if c["task"] == "jc2b" and c["n"] == 1_000_000 and c["init"]}
    out = {}
    for a in CS15B_ARMS:
        for m in models(a):
            run = m.removeprefix("mtx-")
            if run not in cells:
                raise SystemExit(f"FATAL: no JetClass-II 10^6 fine-tuning cell for {run}")
            out.setdefault(a, []).extend([(f"ft-{m}", cells[run]),
                                          (f"lp-{m}", f"{R5.V5['fits_v5_probs']}/{m}/best70/jc2bal_N1000000.npz")])
    return out


def cs15b_spec(tag: str, mem: str) -> str:
    """bs3-boot-cs15b-raunav (not launched before the 10^6 reruns land): accuracy, fine-tuned minus linear probe, per
    arm, the same networks and test jets, SC-1 (paired_bootstrap --task jc2b reads the cells, the .npz as probes)."""
    ins = cs15b_inputs()
    ms = [x for v in ins.values() for x in v]
    diffs = " ".join(f"--diff ft-lp-{a}={','.join(n for n, _ in v if n.startswith('ft-'))}/"
                     f"{','.join(n for n, _ in v if n.startswith('lp-'))}" for a, v in ins.items())
    done = [f"{p}/DONE" for n, p in ms if n.startswith("ft-")] + [p for n, p in ms if n.startswith("lp-")]
    cmd = ("python3 experiments/STATS/paired_bootstrap.py --task jc2b " + " ".join(f"--model {n}={p}" for n, p in ms)
           + f" --headline-only {diffs} --b {B} --out {CS15B_OUT}")
    body = "\n  ".join([BT.check_inputs(done), f"mkdir -p {BOOT}", f"[ -e {CS15B_OUT} ] || " + BT.peak(cmd)])
    return _spec(tag, "bs3-boot-cs15b-raunav", [body], 1, mem, "1",
                 "CS-15b: JetClass-II 10^6 accuracy, fine-tuned against linear probe, per arm (SC-1)")

# ------------------------------------------------------------------ RN1: JetClass linear-probe accuracy, partitions (stored)
JC1_OUT = f"{BOOT}/probe_jc1_N1000000.parts.json"
JC1_DIFFS = (("flavf0", "l188"), ("rand", "l188"), ("rand", "r16q1"))


def jc1_parts_spec(tag: str, mem: str) -> str:
    """bs3-boot-jc1parts-raunav: the JetClass linear probes' stored test probabilities (fits_v5_probs jc1_N1000000.npz,
    the jets of bootstrap/l3_v5/probe_jc1_N1000000.json), the partitions' rows; CPU, nothing fitted."""
    arms = list(dict.fromkeys(x for p in JC1_DIFFS for x in p))
    ins = {m: f"{R5.V5['fits_v5_probs']}/{m}/best70/jc1_N1000000.npz" for a in arms for m in models(a)}
    cmd = ("python3 experiments/STATS/paired_bootstrap.py --task npz " + " ".join(f"--model {m}={p}" for m, p in ins.items())
           + f" --headline {BT.diffs(JC1_DIFFS)} --b {B} --out {JC1_OUT}")
    body = "\n  ".join([BT.check_inputs(ins.values()), f"mkdir -p {BOOT}", f"[ -e {JC1_OUT} ] || " + BT.peak(cmd)])
    return _spec(tag, "bs3-boot-jc1parts-raunav", [body], 1, mem, "1",
                 "RN1: JetClass linear-probe accuracy, the partitions against 188 and 17 classes (stored scores)")

# ------------------------------------------------------------------ SC-2 #1: the transfer-head term on per-jet scores
ANALYTIC_OUT = {f"{BOOT}/analytic_jc2bal_pairs_bc_vs_bq_cs.json":
                lambda m: f"{R5.V5['fits_v5_probs']}/{m}/best70/jc2bal_pairs_bc_vs_bq_cs.npz",
                f"{BOOT}/analytic_top_N1200000.itop4.json":
                lambda m: f"{R5.V5['fits_v5_fill_probs']}/{m}/best70.itop4/top_N1200000.npz"}
ANALYTIC_DIFFS = (("r16q1", "l188"), ("l162", "l188"))


def analytic_spec(tag: str, mem: str) -> str:
    """bs3-analytic-raunav: experiments/STATS/analytic_check.py on two linear-probe rows of each file, eps_S at
    eps_B = 1% with the transfer-head rows' analytic term against the paired bootstrap."""
    lines = [BT.check_inputs([f(m) for f in ANALYTIC_OUT.values() for p in ANALYTIC_DIFFS for a in p for m in models(a)]),
             f"mkdir -p {BOOT}"]
    for out, f in ANALYTIC_OUT.items():
        d = " ".join(f"--diff {BT.dname(a, b)}={','.join(f(m) for m in models(a))}:{','.join(f(m) for m in models(b))}"
                     for a, b in ANALYTIC_DIFFS)
        lines.append(f"[ -e {out} ] || " + BT.peak(f"python3 experiments/STATS/analytic_check.py {d} --b {B} --out {out}"))
    return _spec(tag, "bs3-analytic-raunav", ["\n  ".join(lines)], 1, mem, "1",
                 "SC-2 #1: the transfer-head analytic term against the paired bootstrap on per-jet scores")


# ------------------------------------------------------------------ R5-17: X->ee vs X->mumu at 17 classes (stored scores)
EE_OUT = f"{BOOT}/probe_jc2bal_pairs_ee_vs_mm.r16q1.json"
EE_DIFFS = (("r16q1", "l188"), ("r16q1", "initbn"), ("r16q1", "r42q1"))


def ee_spec(tag: str, mem: str) -> str:
    """bs3-boot-ee-raunav: the 17-class linear probe's X->ee vs X->mumu rows beside the 43-class ones
    (bootstrap/l3_v5/probe_jc2bal_pairs_ee_vs_mm.json), from the stored fits_v5_sib2 scores; CPU, nothing fitted."""
    arms = list(dict.fromkeys(x for p in EE_DIFFS for x in p))
    ins = {m: BT.pair_path("ee_vs_mm", m) for a in arms for m in models(a)}
    cmd = (f"python3 experiments/STATS/paired_bootstrap.py --task npz " + " ".join(f"--model {m}={p}" for m, p in ins.items())
           + f" --headline {BT.diffs(EE_DIFFS)} --b {B} --out {EE_OUT}")
    body = "\n  ".join([BT.check_inputs(ins.values()), f"mkdir -p {BOOT}", f"[ -e {EE_OUT} ] || " + BT.peak(cmd)])
    return _spec(tag, "bs3-boot-ee-raunav", [body], 1, mem, "1",
                 "R5-17: the 17-class X->ee vs X->mumu linear-probe rows from stored scores (CPU)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--what", required=True, choices=("smoke", "smoke-bench", "fit", "boot", "analytic", "ee",
                                                      "mass-helper", "boot-kin", "cs15b", "jc1parts"))
    ap.add_argument("--family", choices=tuple(FAMILY_ARMS))
    ap.add_argument("--part", choices=tuple(BOOT_PARTS), help="with --what boot")
    ap.add_argument("--mem", required=True, help="the pods' memory, 1.5-2x the smoke's peak")
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--skip-smoke", action="store_true", help="leave the smoke's row to the smoke job")
    ap.add_argument("--arms", nargs="+", help="with --what fit: only these arms' runs")
    ap.add_argument("--suffix", default="", help="with --what fit: a suffix to the job name (a later set of arms)")
    a = ap.parse_args(argv)
    if (a.what == "fit") != (a.family is not None) or (a.what == "boot") != (a.part is not None):
        raise SystemExit("FATAL: --family with --what fit and --part with --what boot, and only then")
    name, text = {"smoke": lambda: (SMOKE_NAME["jc2bal"], smoke_spec(a.tag, a.mem)),
                  "smoke-bench": lambda: (SMOKE_NAME["bench"], smoke_spec(a.tag, a.mem, "bench")),
                  "fit": lambda: (f"bs3-mlp-{a.family}{a.suffix}-raunav",
                                  fit_spec(a.tag, a.family, a.mem, a.parallel, a.skip_smoke, a.arms, a.suffix)),
                  "boot": lambda: (f"bs3-boot-{a.part}-raunav", boot_mass_spec(a.tag, a.mem) if a.part == "jc2mass"
                                   else boot_spec(a.tag, a.mem, a.part)),
                  "analytic": lambda: ("bs3-analytic-raunav", analytic_spec(a.tag, a.mem)),
                  "ee": lambda: ("bs3-boot-ee-raunav", ee_spec(a.tag, a.mem)),
                  "mass-helper": lambda: ("bs3-boot-jc2mass-helper-raunav", boot_mass_helper_spec(a.tag, a.mem)),
                  "boot-kin": lambda: ("bs3-boot-kin-raunav", boot_kin_spec(a.tag, a.mem)),
                  "cs15b": lambda: ("bs3-boot-cs15b-raunav", cs15b_spec(a.tag, a.mem)),
                  "jc1parts": lambda: ("bs3-boot-jc1parts-raunav", jc1_parts_spec(a.tag, a.mem))}[a.what]()
    p = OUT_DIR / f"job-{name}.yaml"
    p.write_text(text)
    print(f"wrote {p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
