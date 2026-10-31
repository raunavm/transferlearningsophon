#!/usr/bin/env python3
"""CPU jobs that read the stored per-jet scores once their inputs exist (plan 2026-10-29, Step 1.7
and 1.10), and the status CronJob. Specs in experiments/FT/k8s/curves/ and
experiments/MTX/k8s/v2/status/.

  curves-ft-raunav      experiments/FT/ft_curves.py cells on the JetClass, top and q/g trees, the
                        pretrained fine-tunes (best70) and the V100 from-scratch references
  curves-jc2-raunav     ... on the 188-class JetClass-II trees (leg1_188)
  curves-full-raunav    ft_curves.py cells at the full training sets only, all 27 paper runs and the
                        from-scratch seeds (PIN_FULL)
  curves-probe-raunav   ft_curves.py probes on the probe refit's saved test outputs (fits_v3_probs)
  boot-ft-raunav        experiments/STATS/paired_bootstrap.py per task and training size on the 16
                        runs reported for fine-tuning and the three from-scratch seeds on subset s1
  boot-jc2-raunav       ... on JetClass-II at 188 classes (macro AUC)
  boot-probe-raunav     ... on the 30 probes: top, q/g, Herwig at every size; JetClass and
                        JetClass-II (macro AUC) and the six pairs at the largest
  status-raunav         CronJob, experiments/MTX/status_v2.py every 30 min

Ratios: each run over the 188-class run of its index, a mass-output run also over its run without
the mass output, a flavour run over the runs of its index in FLAV_OVER, and each run over its
baseline -- the from-scratch seed of its index trained on the pretrained fine-tunes' subset s1
(scratch-s1-<k>) for fine-tuning, the untrained trunk of its index for probes. Every call of a job
reports every ratio (the probe's Herwig and pair tasks, the fine-tunes' Herwig cells included).

Each job polls every 30 min until all its inputs exist (the cells' DONE; a probe refit's
<model>.json, written last), gives up after 6 days with exit 42 (the pod failure policy fails the
Job at once), and writes only new files: ft_curves.py one JSON per tree into CURVES, each written
whole, so a retried pod skips the trees already written and does the rest; the bootstrap jobs and
curves-probe refuse to start if their output exists, the bootstrap jobs writing a directory of their
own under BOOT, moved into place whole.

    python3 scripts/build_curve_jobs.py --pin-not-yet-tagged [--check-only] [--repin]
"""
from __future__ import annotations

import argparse
import importlib.util
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "experiments" / "FT" / "k8s" / "curves"
STATUS_DIR = ROOT / "experiments" / "MTX" / "k8s" / "v2" / "status"
# mtx-s2.18 (4f92021, the linear-probe refit) was tagged before this builder was committed;
# mtx-s2.19 carried the first specs, none of the curves or bootstrap jobs launched at it
PIN = "mtx-s2.20"
PIN_FULL = "mtx-s2.28"         # curves-full (2026-10-31): ft_curves.py --n
STATUS_PIN = "mtx-s2.19"       # the status CronJob, launched at it (RUNS.csv status-raunav, 2026-10-30)
NEEDED = {"experiments/FT/ft_curves.py": "def roc_points",
          "experiments/STATS/paired_bootstrap.py": "P.AucScorer",
          "experiments/MTX/status_v2.py": "def status"}


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


BX = _load("build_extract_jobs", "scripts/build_extract_jobs.py")
BF = _load("build_ft_jobs", "scripts/build_ft_jobs.py")

V2, REF = BF.V2_ROOT, BF.V2_REF_ROOT
CURVES = f"{V2}/curves"
BOOT = "/data/results/bootstrap"
LP_ROOT = "/data/results/eval/v2_linprobe"            # scripts/build_linprobe_jobs.py LP_ROOT
FITS_V3, PROBS_V3 = f"{LP_ROOT}/fits_v3", f"{LP_ROOT}/fits_v3_probs"
LEG1_188 = BF.LEG188_TREE                              # the 188-class leg's tree
SCRATCH, SEEDS = BF.SCRATCH_REF, BF.INITS_LATER[BF.SCRATCH_REF][0][3]
# The fine-tunes' baseline: from-scratch seeds 1-3 on subset s1, the pretrained fine-tunes' subset,
# named scratch-s1-<seed>. Seed 1 is the reference's own; seeds 2-3 are REF_S1's (the reference's
# seed S trains on subset sS off JetClass-II) at the trees and sizes build_ft_jobs trains there
# (REF_S1_TREES), else the reference's own (s1_tree).
BASELINE = "scratch-s1"
REF_S1, REF_S1_TREES = BF.V2_REF_S1_ROOT, BF.REF_S1_TREES   # scripts/build_ft_jobs.py --v2-extra
FT_RUNS = list(BF.V2_LEG188_RUNS)                      # the 16 runs reported for fine-tuning
PROBE_MODELS = [(r, "best70") for r in BF.v2_paper_runs()] + [(i, "init") for i, _ in BX.v2_init_refs()]
_LPV = _load("linear_probe_v2", "experiments/EVAL/linear_probe_v2.py")
PAIRS_188 = (*_LPV.PAIRS, *_LPV.QCD_PAIRS)                # the probe's jc2_188_pairs
NMAX = BF.SIZES[-1]
SIZES = {"jc1": BF.SIZES, "jc2_188": BF.SIZES, "top": BF.BENCH_SIZES["top"],
         "qg": BF.BENCH_SIZES["qg"], "qg_herwig": BF.BENCH_SIZES["qg"]}
# output name: (tree, ft_curves task, the bootstrap's tasks, its init directories, their seeds)
TREES = {
    "best70_leg2": (f"{V2}/best70/leg2", "jc1", ("jc1",), FT_RUNS, [1]),
    "best70_leg_top": (f"{V2}/best70/leg_top", "top", ("top",), FT_RUNS, [1]),
    "best70_leg_qg": (f"{V2}/best70/leg_qg", "qg", ("qg", "qg_herwig"), FT_RUNS, [1]),
    "ref_v100_jc1_leg2": (f"{REF}/jc1/leg2", "jc1", ("jc1",), [SCRATCH], SEEDS),
    "ref_v100_bench_leg_top": (f"{REF}/bench/leg_top", "top", ("top",), [SCRATCH], SEEDS),
    "ref_v100_bench_leg_qg": (f"{REF}/bench/leg_qg", "qg", ("qg", "qg_herwig"), [SCRATCH], SEEDS),
    f"best70_{LEG1_188}": (f"{V2}/best70/{LEG1_188}", "jc2_188", ("jc2_188",), FT_RUNS, [1]),
    f"ref_v100_jc2_{LEG1_188}": (f"{REF}/jc2/{LEG1_188}", "jc2_188", ("jc2_188",), [SCRATCH], SEEDS),
}
# the pretrained tree and its from-scratch reference, per job
FAMILIES = {"ft": (("best70_leg2", "ref_v100_jc1_leg2"), ("best70_leg_top", "ref_v100_bench_leg_top"),
                   ("best70_leg_qg", "ref_v100_bench_leg_qg")),
            "jc2": ((f"best70_{LEG1_188}", f"ref_v100_jc2_{LEG1_188}"),)}
# a flavour run over these runs of its index
FLAV_OVER = {"flavf0": ("r16q1",), "flavf1": ("flavf0",), "flavf1r": ("flavf1", "flavf0")}
POLL_S, GIVE_UP_S = 1800, 6 * 86400
# curves-full: the full training sets only (plan 2026-10-30: below them the fixed-budget recipe
# replaces these trees), every paper run and the three from-scratch seeds; (tree, task, N)
FULL_RUNS = [r.removeprefix("mtx-") for r in BF.v2_paper_runs()]
FULL = {"best70_leg2": (f"{V2}/best70/leg2", "jc1", NMAX, FULL_RUNS, [1]),
        "best70_leg_top": (f"{V2}/best70/leg_top", "top", BF.BENCH_SIZES["top"][-1], FULL_RUNS, [1]),
        "best70_leg_qg": (f"{V2}/best70/leg_qg", "qg", BF.BENCH_SIZES["qg"][-1], FULL_RUNS, [1]),
        "ref_v100_jc1_leg2": (f"{REF}/jc1/leg2", "jc1", NMAX, [SCRATCH], SEEDS),
        "ref_v100_bench_leg_top": (f"{REF}/bench/leg_top", "top", BF.BENCH_SIZES["top"][-1], [SCRATCH], SEEDS),
        "ref_v100_bench_leg_qg": (f"{REF}/bench/leg_qg", "qg", BF.BENCH_SIZES["qg"][-1], [SCRATCH], SEEDS)}


def _alt(xs) -> str:
    """A bash brace alternative; a single item unbraced (bash leaves "{a}" as it is)."""
    xs = [str(x) for x in xs]
    return xs[0] if len(xs) == 1 else "{" + ",".join(xs) + "}"


def done_markers(name: str) -> str:
    tree, _, _, inits, seeds = TREES[name]
    return f"{tree}/{_alt(inits)}/N{_alt(SIZES[TREES[name][1]])}/s{_alt(seeds)}/DONE"


def s1_tree(ref: str, n: int) -> str:
    """The tree of the from-scratch seeds 2-3 on subset s1 for reference tree ref at N=n: its twin
    under REF_S1 where build_ft_jobs trains them (REF_S1_TREES), else the reference's own (on
    JetClass-II every seed trains on s1; at the full top and q/g sets every subset is the same jets)."""
    tree = TREES[ref][0]
    part, leaf = tree.removeprefix(REF + "/").rsplit("/", 1)
    if leaf in REF_S1_TREES and n in REF_S1_TREES[leaf][1]:
        assert part == REF_S1_TREES[leaf][0], (tree, REF_S1_TREES[leaf])
        return f"{REF_S1}/{part}/{leaf}"
    return tree


def s1_markers(ref: str) -> list[str]:
    """The DONE markers of the REF_S1 cells reference tree ref's baseline reads."""
    ns = [n for n in SIZES[TREES[ref][1]] if s1_tree(ref, n) != TREES[ref][0]]
    return [f"{s1_tree(ref, ns[0])}/{SCRATCH}/N{_alt(ns)}/s{_alt(BF.REF_S1_SEEDS)}/DONE"] if ns else []


def ratio_pairs(runs, prefix: str, baseline) -> list[str]:
    """A/B for each run: over the 188-class run of its index, a mass-output run also over its run
    without the mass output, a flavour run over its FLAV_OVER runs of its index, and over
    baseline(index)."""
    out = []
    for r in runs:
        base, k = r.rsplit("-s", 1)
        over = (([] if base == "l188" else ["l188"]) + ([base.removesuffix("mass")] if base.endswith("mass") else [])
                + list(FLAV_OVER.get(base, ())))
        assert all(f"{b}-s{k}" in runs for b in over), (r, over)
        out += [f"{prefix}{r}/{prefix}{b}-s{k}" for b in over] + [f"{prefix}{r}/{baseline(k)}"]
    return out


def _guard(outs: list[str], inputs: list[str]) -> str:
    return (("          # every output new; then wait in the pod until every input exists\n"
             f"          for f in {' '.join(outs)}; do\n"
             '            [ ! -e "${f}" ] || { echo "FATAL: ${f} exists; use a new output"; exit 42; }\n'
             "          done\n" if outs else "          # wait in the pod until every input exists\n")
            + "          missing () { local n=0 f; for f in " + " ".join(inputs) + "; do\n"
            '            [ -e "${f}" ] || n=$((n+1)); done; echo ${n}; }\n'
            "          T0=$(date +%s)\n"
            '          while M=$(missing); [ "${M}" -gt 0 ]; do\n'
            f'            [ $(( $(date +%s) - T0 )) -lt {GIVE_UP_S} ] || '
            '{ echo "FATAL: ${M} inputs still missing after 6 days"; exit 42; }\n'
            f'            echo "$(date -u +%FT%TZ) waiting for ${{M}} inputs"; sleep {POLL_S}\n'
            "          done\n")


def _boot(out: str, pairs: list[str], fn: str, calls: list[str]) -> str:
    """Write into a staging directory of this attempt, moved into place whole; fn defines the
    bash function each call runs, PAIRS the ratios every call reports."""
    return ("          TMP=" + out + ".staging.$(date -u +%s)\n          mkdir -p ${TMP}\n"
            + "          PAIRS=(" + " ".join(f"--pair {q}" for q in pairs) + ")\n" + fn
            + "".join(f"          {c}\n" for c in calls)
            + f'          [ ! -e {out} ] || {{ echo "FATAL: {out} appeared while this ran"; exit 42; }}\n'
            f"          mv ${{TMP}} {out}\n          ls -la {out}\n")


# boot TASK PRETRAINED_TREE REFERENCE_TREE N S1_TREE: the reported runs' cells at N and the
# from-scratch seeds on subset s1, seed 1 in REFERENCE_TREE and seeds 2-3 in S1_TREE (s1_tree)
BOOT_FT = """          RUNS="{runs}"
          boot () {{
            local args=() r s
            for r in ${{RUNS}}; do args+=(--model "${{r}}=$2/${{r}}/N$4/s1"); done
            args+=(--model "{base}-1=$3/{scratch}/N$4/s1")
            for s in {seeds}; do args+=(--model "{base}-${{s}}=$5/{scratch}/N$4/s${{s}}"); done
            python3 experiments/STATS/paired_bootstrap.py --task $1 "${{args[@]}}" "${{PAIRS[@]}}" \\
              --out ${{TMP}}/$1_N$4.json || halt
          }}
"""
# boot DATASET N: every probe's saved outputs <model>/<checkpoint>/<dataset>_N<N>.npz
BOOT_PROBE = """          MODELS="{models}"
          boot () {{
            local args=() m
            for m in ${{MODELS}}; do args+=(--model "${{m%%:*}}={probs}/${{m%%:*}}/${{m##*:}}/$1_N$2.npz"); done
            python3 experiments/STATS/paired_bootstrap.py --task npz "${{args[@]}}" "${{PAIRS[@]}}" \\
              --out ${{TMP}}/$1_N$2.json || halt
          }}
"""


def curves_body(names: list[str]) -> str:
    outs = [f"{CURVES}/{n}.json" for n in names]
    # no up-front refusal: each tree's JSON is written whole, so a pod retried after an eviction
    # skips the trees its first attempt wrote (ft_curves.py itself never overwrites)
    return _guard([], [done_markers(n) for n in names]) + "".join(
        f'          if [ -e {o} ]; then echo "skip {o}: exists"; else\n'
        f"            python3 experiments/FT/ft_curves.py cells --tree {TREES[n][0]} --task {TREES[n][1]} \\\n"
        f"              --out {o} || halt\n          fi\n" for n, o in zip(names, outs))


def curves_full_body() -> str:
    markers = [f"{t}/{_alt(inits)}/N{n}/s{_alt(seeds)}/DONE" for t, _, n, inits, seeds in FULL.values()]
    return _guard([], markers) + "".join(
        f'          if [ -e {CURVES}/full_{name}.json ]; then echo "skip full_{name}.json: exists"; else\n'
        f"            python3 experiments/FT/ft_curves.py cells --tree {t} --task {task} --n {n} \\\n"
        f"              --out {CURVES}/full_{name}.json || halt\n          fi\n"
        for name, (t, task, n, _, _) in FULL.items())


def curves_probe_body() -> str:
    out = f"{CURVES}/linprobe_fits_v3.json"
    return _guard([out], [f"{FITS_V3}/{_alt([m for m, _ in PROBE_MODELS])}.json"]) + (
        f"          python3 experiments/FT/ft_curves.py probes --probs-root {PROBS_V3} \\\n"
        f"            --models {' '.join(m for m, _ in PROBE_MODELS)} --out {out} || halt\n")


def boot_ft_body(family: str) -> str:
    out = f"{BOOT}/{family}"
    assert SEEDS[0] == 1 and tuple(SEEDS[1:]) == BF.REF_S1_SEEDS, (SEEDS, BF.REF_S1_SEEDS)
    fn = BOOT_FT.format(runs=" ".join(FT_RUNS), seeds=" ".join(map(str, SEEDS[1:])), scratch=SCRATCH, base=BASELINE)
    calls = [f"boot {task} {TREES[pre][0]} {TREES[ref][0]} {n} {s1_tree(ref, n)}"
             for pre, ref in FAMILIES[family] for task in TREES[pre][2] for n in SIZES[task]]
    inputs = [done_markers(t) for pair in FAMILIES[family] for t in pair]
    inputs += [m for _, ref in FAMILIES[family] for m in s1_markers(ref)]
    return _guard([out], inputs) + _boot(out, ratio_pairs(FT_RUNS, "", lambda k: f"{BASELINE}-{k}"), fn, calls)


def boot_probe_body() -> str:
    out = f"{BOOT}/probe"
    pairs = ratio_pairs([m.removeprefix("mtx-") for m, _ in PROBE_MODELS if m.startswith("mtx-")],
                        "mtx-", lambda k: f"init-s{k}")
    fn = BOOT_PROBE.format(models=" ".join(f"{m}:{ck}" for m, ck in PROBE_MODELS), probs=PROBS_V3)
    calls = [f"boot {ds} {n}" for ds in ("top", "qg", "qg_herwig") for n in SIZES[ds]]
    calls += [f"boot {ds} {NMAX}" for ds in ("jc1", "jc2_188")] + [f"boot jc2_188_pairs_{p} {NMAX}" for p in PAIRS_188]
    return _guard([out], [f"{FITS_V3}/{_alt([m for m, _ in PROBE_MODELS])}.json"]) + _boot(out, pairs, fn, calls)


def cpu_job(name: str, body: str, what: str, pin: str) -> str:
    """build_extract_jobs' CPU template (as build_linprobe_jobs' MLP probe jobs), backoffLimit 1."""
    text = BX.V1ERR_TEMPLATE.format(name=name, image=BX.IMAGE, pin=pin, body=body, mem="16Gi", cpu="4",
                                    gpu_req="", gpu_check="", node_exclude=BX.NODE_EXCLUDE)
    header = ("  # GENERATED by scripts/build_curve_jobs.py -- do not hand-edit. Regenerate.\n"
              f"  # {what}. CPU; waits in the pod for its inputs (30 min polls, 6 days, then exit 42).\n")
    return BF._derive(text, [("metadata:\n", "metadata:\n" + header, 1),
                             ("  backoffLimit: 6\n", "  backoffLimit: 1\n", 1)], name)


def status_spec(pin: str) -> str:
    """The backup CronJob's spec (build_mtx_launch.v2_backup_spec: image, volume, region, pinned
    clone, every 30 min, never retried) running status_v2.py instead."""
    _, text = BF._mtx_launch().v2_backup_spec(pin)
    return BF._derive(text, [
        ("mtx2-backup-raunav", "status-raunav", 1),
        ("  # v2 BACKUPS: experiments/MTX/backup_v2_resume.py every 30 min, CPU only.\n"
         "  # GENERATED by scripts/build_mtx_launch.py (v2_backup_spec) -- do not hand-edit.\n",
         "  # STATUS: experiments/MTX/status_v2.py every 30 min, CPU only, no cluster API: counts the jobs'\n"
         "  # completion files into /data/results/status/status.json (a few kB).\n"
         "  # GENERATED by scripts/build_curve_jobs.py (from build_mtx_launch.v2_backup_spec) -- do not hand-edit.\n", 1),
        ('              USED=$(df --output=pcent /data | tail -1 | tr -dc 0-9)\n'
         '              [ "${USED}" -lt 85 ] || { echo "FATAL: /data at ${USED}%, no backups above 85%"; exit 1; }\n',
         "", 1),
        ("python3 experiments/MTX/backup_v2_resume.py --root /data/results/mtx_v2",
         "python3 experiments/MTX/status_v2.py --out /data/results/status/status.json", 1),
        ('memory: "8Gi", cpu: "2"', 'memory: "2Gi", cpu: "1"', 2),
    ], "status CronJob")


def build(pin: str) -> dict[pathlib.Path, str]:
    assert {f"mtx-{r}" for r in FT_RUNS} <= set(BF.v2_paper_runs()), FT_RUNS
    jobs = {
        "curves-ft": (curves_body([n for pair in FAMILIES["ft"] for n in pair]),
                      "ROC points and per-class tables, JetClass, top and q/g fine-tunes"),
        "curves-jc2": (curves_body([n for pair in FAMILIES["jc2"] for n in pair]),
                       "ROC points per class, JetClass-II 188-class fine-tunes"),
        "curves-probe": (curves_probe_body(), "ROC points and per-class tables, the linear probes (fits_v3)"),
        "boot-ft": (boot_ft_body("ft"), "Paired bootstrap over test jets, JetClass, top and q/g fine-tunes"),
        "boot-jc2": (boot_ft_body("jc2"), "Paired bootstrap over test jets, JetClass-II 188-class fine-tunes"),
        "boot-probe": (boot_probe_body(), "Paired bootstrap over test jets, the linear probes (fits_v3)"),
    }
    out = {OUT_DIR / f"job-{j}-raunav.yaml": cpu_job(f"{j}-raunav", body, what, pin)
           for j, (body, what) in jobs.items()}
    out[OUT_DIR / "job-curves-full-raunav.yaml"] = cpu_job(
        "curves-full-raunav", curves_full_body(),
        "ROC points and per-class tables, JetClass, top and q/g fine-tunes on the full training sets", PIN_FULL)
    out[STATUS_DIR / "cronjob-status-raunav.yaml"] = status_spec(STATUS_PIN)
    return out


def _pin_of(text: str) -> str | None:
    m = re.search(r'--branch "([^"$]+)"|REPO_REF\n\s+value: "([^"]+)"', text)
    return m and (m.group(1) or m.group(2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pin", default=PIN)
    ap.add_argument("--pin-not-yet-tagged", action="store_true",
                    help="the tag is created after the commit: check the working tree instead")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--repin", action="store_true", help="allow moving the pin of a spec on disk")
    a = ap.parse_args()
    BF.verify_pin(a.pin, a.pin_not_yet_tagged, NEEDED)
    specs = build(a.pin)
    moved = [f"  {p.relative_to(ROOT)}: {_pin_of(p.read_text())} -> {_pin_of(t)}" for p, t in specs.items()
             if p.exists() and _pin_of(p.read_text()) != _pin_of(t)]
    if moved and not a.repin and not a.check_only:
        sys.exit("FATAL: this would move the pin of specs whose jobs may have run:\n" + "\n".join(moved))
    for p, text in specs.items():
        if not a.check_only:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
        print(f"{p.relative_to(ROOT)} {'checked' if a.check_only else 'written'} (pin {_pin_of(text)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
