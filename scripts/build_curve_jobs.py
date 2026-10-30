#!/usr/bin/env python3
"""CPU jobs that read the stored per-jet scores once their inputs exist (plan 2026-10-29, Step 1.7
and 1.10), and the status CronJob. Specs in experiments/FT/k8s/curves/ and
experiments/MTX/k8s/v2/status/.

  curves-ft-raunav      experiments/FT/ft_curves.py cells on the JetClass, top and q/g trees, the
                        pretrained fine-tunes (best70) and the V100 from-scratch references
  curves-jc2-raunav     ... on the 188-class JetClass-II trees (leg1_188)
  curves-probe-raunav   ft_curves.py probes on the probe refit's saved test outputs (fits_v3_probs)
  boot-ft-raunav        experiments/STATS/paired_bootstrap.py per task and training size on the 16
                        runs reported for fine-tuning and the three from-scratch references
  boot-jc2-raunav       ... on JetClass-II at 188 classes (macro AUC)
  boot-probe-raunav     ... on the 30 probes: top, q/g, Herwig at every size; JetClass and
                        JetClass-II (macro AUC) and the six pairs at the largest
  status-raunav         CronJob, experiments/MTX/status_v2.py every 30 min

Ratios: each run over the 188-class run of its index, a mass-output run also over its run without
the mass output, and each run over its baseline -- the from-scratch reference trained on the same
subset (s1) for fine-tuning, the untrained trunk of its index for probes.

Each job polls every 30 min until all its inputs exist (the cells' DONE; a probe refit's
<model>.json, written last), gives up after 6 days with exit 42 (the pod failure policy fails the
Job at once), and writes only new files: ft_curves.py one JSON per tree into CURVES, each written
whole, so a retried pod skips the trees already written and does the rest; the bootstrap jobs and
curves-probe refuse to start if their output exists, the bootstrap jobs writing a directory of their
own under BOOT, moved into place whole.

    python3 scripts/build_curve_jobs.py --pin-not-yet-tagged [--check-only]
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
# mtx-s2.18 (4f92021, the linear-probe refit) was tagged before this builder was committed
PIN = "mtx-s2.19"
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
BASELINE = f"{SCRATCH}-s1"     # the from-scratch reference on the pretrained fine-tunes' subset
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
POLL_S, GIVE_UP_S = 1800, 6 * 86400


def _alt(xs) -> str:
    """A bash brace alternative; a single item unbraced (bash leaves "{a}" as it is)."""
    xs = [str(x) for x in xs]
    return xs[0] if len(xs) == 1 else "{" + ",".join(xs) + "}"


def done_markers(name: str) -> str:
    tree, _, _, inits, seeds = TREES[name]
    return f"{tree}/{_alt(inits)}/N{_alt(SIZES[TREES[name][1]])}/s{_alt(seeds)}/DONE"


def ratio_pairs(runs, prefix: str, baseline) -> list[str]:
    """A/B for each run: over the 188-class run of its index, a mass-output run also over its run
    without the mass output, and over baseline(index)."""
    out = []
    for r in runs:
        base, k = r.rsplit("-s", 1)
        if base != "l188":
            out.append(f"{prefix}{r}/{prefix}l188-s{k}")
        if base.endswith("mass"):
            out.append(f"{prefix}{r}/{prefix}{base.removesuffix('mass')}-s{k}")
        out.append(f"{prefix}{r}/{baseline(k)}")
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


# boot TASK PRETRAINED_TREE REFERENCE_TREE N: the reported runs' cells and the references' at N
BOOT_FT = """          RUNS="{runs}"
          boot () {{
            local args=() r s
            for r in ${{RUNS}}; do args+=(--model "${{r}}=$2/${{r}}/N$4/s1"); done
            for s in {seeds}; do args+=(--model "{scratch}-s${{s}}=$3/{scratch}/N$4/s${{s}}"); done
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


def curves_probe_body() -> str:
    out = f"{CURVES}/linprobe_fits_v3.json"
    return _guard([out], [f"{FITS_V3}/{_alt([m for m, _ in PROBE_MODELS])}.json"]) + (
        f"          python3 experiments/FT/ft_curves.py probes --probs-root {PROBS_V3} \\\n"
        f"            --models {' '.join(m for m, _ in PROBE_MODELS)} --out {out} || halt\n")


def boot_ft_body(family: str) -> str:
    out = f"{BOOT}/{family}"
    fn = BOOT_FT.format(runs=" ".join(FT_RUNS), seeds=" ".join(map(str, SEEDS)), scratch=SCRATCH)
    calls = [f"boot {task} {TREES[pre][0]} {TREES[ref][0]} {n}"
             for pre, ref in FAMILIES[family] for task in TREES[pre][2] for n in SIZES[task]]
    inputs = [done_markers(t) for pair in FAMILIES[family] for t in pair]
    return _guard([out], inputs) + _boot(out, ratio_pairs(FT_RUNS, "", lambda k: BASELINE), fn, calls)


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
    out[STATUS_DIR / "cronjob-status-raunav.yaml"] = status_spec(pin)
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
    moved = [f"  {p.relative_to(ROOT)}: {_pin_of(p.read_text())} -> {a.pin}" for p in specs
             if p.exists() and _pin_of(p.read_text()) != a.pin]
    if moved and not a.repin and not a.check_only:
        sys.exit("FATAL: this would move the pin of specs whose jobs may have run:\n" + "\n".join(moved))
    for p, text in specs.items():
        if not a.check_only:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
        print(f"{p.relative_to(ROOT)} {'checked' if a.check_only else 'written'} (pin {a.pin})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
