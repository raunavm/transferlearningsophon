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
FAMILY_ARMS = {"jc2bal": ("l188", "l162"), "coarse": ("l188", "l162", "r42q1", "r16q1"), "bench": tuple(ARMS)}
ROWS = [(f, f"mtx-{r}") for f, arms in FAMILY_ARMS.items() for a in arms for r in ARMS[a]]
# smokes: jc2bal, the largest output and a bit-for-bit check against fits_v5_mlp; bench, the largest training set
SMOKES = {"jc2bal": ("jc2bal", "mtx-l188-s1"), "bench": ("bench", "mtx-l188-s1")}
SMOKE_NAME = {"jc2bal": "bs3-mlp-smoke-raunav", "bench": "bs3-mlp-smoke-bench-raunav"}
FILES = {"jc2bal": ("jc2bal_N1000000",), "coarse": ("jc2bal_R16_Q1_N1000000",),
         "bench": ("top_N1200000", "qg_N1600000", "qg_herwig_N1600000")}
# the bootstraps: output -> (family, npz name, the differences (first arm, second arm), flags)
BENCH_DIFFS = tuple((a, "l188") for a in ARMS if a != "l188")
BOOT_OUT = {f"{BOOT}/mlp_jc2bal_N1000000.json": ("jc2bal", "jc2bal_N1000000", (("l162", "l188"),), "--headline-only"),
            f"{BOOT}/mlp_jc2bal_R16_Q1_N1000000.json": ("coarse", "jc2bal_R16_Q1_N1000000",
                                                        (("l162", "l188"), ("r42q1", "l188"), ("r16q1", "l188")),
                                                        "--headline-only"),
            **{f"{BOOT}/mlp_{n}.json": ("bench", n, BENCH_DIFFS, "--headline") for n in FILES["bench"]}}


def models(arm: str) -> list[str]:
    return [f"mtx-{r}" for r in ARMS[arm]]


# bench: the linear probes' substitution of the missing inputs (mlp_scores.FILL; third review of 5f4e1b8, B3(c))
CKPT = {"top_N1200000": "best70.itop4", "qg_N1600000": "best70.nqg", "qg_herwig_N1600000": "best70.nqg"}


def npz(family: str, model: str, name: str) -> str:
    return f"{OUT}/{family}/probs/{model}/{CKPT.get(name, 'best70')}/{name}.npz"


def guard() -> str:
    return ('USED=$(df --output=pcent /data | tail -1 | tr -dc 0-9)\n'
            f'[ "${{USED}}" -le {LINE} ] || {{ echo "FATAL: /data at ${{USED}}%"; exit 42; }}')


def fit_row(family: str, model: str) -> str:
    f = f"{OUT}/{family}/{model}.json"
    return (f'[ -f {f} ] && {{ echo "{model} done"; exit 0; }}; ' + BT.peak(
        f"python3 experiments/EVAL/mlp_scores.py --family {family} --model {model} --out {OUT}/{family}"))


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


def fit_rows(family: str, skip_smoke: bool) -> list[tuple[str, str]]:
    """The family's rows; with skip_smoke, without the row a smoke job fits (SMOKES)."""
    return [r for r in ROWS if r[0] == family and not (skip_smoke and r in SMOKES.values())]


def fit_spec(tag: str, family: str, mem: str, parallelism: int, skip_smoke: bool) -> str:
    """bs3-mlp-<family>-raunav: one job a family, so each family's pods are sized to its own measured peak."""
    rows = [fit_row(*r) for r in fit_rows(family, skip_smoke)]
    return _spec(tag, f"bs3-mlp-{family}-raunav", rows, parallelism, mem, THREADS,
                 f"MLP probes with per-jet scores ({family}) for the main text's linear-probe nulls (PI 2026-11-23, DL-9)")


def smoke_spec(tag: str, mem: str, which: str = "jc2bal") -> str:
    return _spec(tag, SMOKE_NAME[which], [fit_row(*SMOKES[which])], 1, mem, THREADS,
                 "MLP probes with per-jet scores, smoke: one fit, its peak memory and seconds")


# the bootstrap jobs: JetClass-II (jc2bal and coarse) and bench, each launched once its fits exist
BOOT_PARTS = {"jc2": ("jc2bal", "coarse"), "bench": ("bench",)}


def boot_outs(part: str) -> dict:
    return {o: v for o, v in BOOT_OUT.items() if v[0] in BOOT_PARTS[part]}


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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--what", required=True, choices=("smoke", "smoke-bench", "fit", "boot", "analytic"))
    ap.add_argument("--family", choices=tuple(FAMILY_ARMS))
    ap.add_argument("--part", choices=tuple(BOOT_PARTS), help="with --what boot")
    ap.add_argument("--mem", required=True, help="the pods' memory, 1.5-2x the smoke's peak")
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--skip-smoke", action="store_true", help="leave the smoke's row to the smoke job")
    a = ap.parse_args(argv)
    if (a.what == "fit") != (a.family is not None) or (a.what == "boot") != (a.part is not None):
        raise SystemExit("FATAL: --family with --what fit and --part with --what boot, and only then")
    name, text = {"smoke": lambda: (SMOKE_NAME["jc2bal"], smoke_spec(a.tag, a.mem)),
                  "smoke-bench": lambda: (SMOKE_NAME["bench"], smoke_spec(a.tag, a.mem, "bench")),
                  "fit": lambda: (f"bs3-mlp-{a.family}-raunav", fit_spec(a.tag, a.family, a.mem, a.parallel, a.skip_smoke)),
                  "boot": lambda: (f"bs3-boot-{a.part}-raunav", boot_spec(a.tag, a.mem, a.part)),
                  "analytic": lambda: ("bs3-analytic-raunav", analytic_spec(a.tag, a.mem))}[a.what]()
    p = OUT_DIR / f"job-{name}.yaml"
    p.write_text(text)
    print(f"wrote {p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
