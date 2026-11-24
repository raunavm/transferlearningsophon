#!/usr/bin/env python3
"""The paired test-jet bootstraps of tab:tests' untested rows (PI 2026-11-23, LEDGER SP-12 step 4: "Run bootstraps only
for the untested comparisons that survive"; SC-1 (1): sigma^2 = each side's run SD^2 + the paired test-jet bootstrap SD^2)
whose per-jet inputs exist now: experiments/STATS/paired_bootstrap.py and experiments/EVAL/mass_sd_boot.py on stored
outputs only, nothing fitted or trained. Every output is a new file beside the L3 files, with new difference names; a
file that exists is skipped, so a retried pod does only what is missing. CPU, every pod on one CPU model
(build_refit_v5_jobs.pin_cpu: the EPYC 7402P nodes; the 7742 nodes beside the volume had no free CPU on 2026-11-23),
at most PARALLEL pods at once. One Indexed Job, boot-tests-raunav:

  0-2  the class-balanced JetClass-II linear probes at 1e6 (fits_v5_probs) of the 23 networks the JetClass-II rows
       compare, split over three pods into CACHE (paired_bootstrap --cache-only): accuracy, the macro AUC and its
       resonant and QCD groups, on 903,922 test jets each
  3    every other file: the class pairs, the 17 merged groups, the smaller training sets (fits_v4_probs: the stored
       class probabilities below 1e6; same C as the v5 fits), the substitutions of the missing inputs, the fine-tuned
       top (against from scratch) and quark/gluon (Herwig over Pythia), and mass_sd_boot_v4.json
  4    waits for the 23 cache files, then probe_jc2bal_N1000000.all.json from them
boot-tests-herwig-raunav (one pod, --herwig): R30 on Herwig minus on Pythia jets, unpaired, into HERWIG_OUT.
boot-tests-smoke-raunav (one pod, --smoke): every input checked, one row (X->bb vs cc, 43 - 188 classes) into SMOKE, and
one JetClass-II network into SMOKE's cache with its peak memory and seconds, to size the job.

    python3 scripts/build_boot_tests_jobs.py --tag mtx-sN.NN [--smoke] [--mem 8Gi]
"""
from __future__ import annotations

import argparse
import importlib.util
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "experiments" / "STATS" / "k8s"


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


R5 = _load("build_refit_v5_jobs", "scripts/build_refit_v5_jobs.py")
BL = _load("build_boot_l3_jobs", "scripts/build_boot_l3_jobs.py")
LP = R5.LP                                                 # /data/results/eval/v2_linprobe_bal
L3, L3_V5 = BL.OUT, R5.L3_V5                               # bootstrap/l3 (fine-tuned), bootstrap/l3_v5 (probes)
FT = BL.FT                                                 # /data/results/ft_v2
CACHE = "/data/results/bootstrap_cache/tests_jc2bal_N1000000"     # outside bootstrap/: never pulled
SMOKE = "/data/results/bootstrap_tests_smoke"
MASS_V3 = R5.MASS_SD_BOOT
MASS_V4 = MASS_V3.replace("_v3.json", "_v4.json")
CPU_MODEL = "AMD EPYC 7402P 24-Core Processor"            # one of R5.CPU_NODES
PARALLEL, CPU = 4, "1"
B = BL.REPLICAS
ARMS = {**R5.BOOTALL_ARMS, "init": ("init-s1", "init-s2", "init-s3"), "initbn": ("initbn-s1", "initbn-s2", "initbn-s3")}


def models(arm: str) -> list[str]:
    return [r if arm in ("init", "initbn") else f"mtx-{r}" for r in ARMS[arm]]


def dname(a: str, b: str) -> str:
    return f"{a}-{'188' if b == 'l188' else b}"


def diffs(pairs) -> str:
    return " ".join(f"--diff {dname(a, b)}={','.join(models(a))}/{','.join(models(b))}" for a, b in pairs)


def pb(out: str, model_args: list[str], flags: str, task: str = "npz") -> str:
    """One paired_bootstrap.py call into out, skipped when out exists."""
    return (f"[ -e {out} ] || python3 experiments/STATS/paired_bootstrap.py --task {task} {' '.join(model_args)} "
            f"{flags} --b {B} --out {out}")


# ------------------------------------------------------------------ JetClass-II, linear probe at 1e6 (Table 1's cell)
JC2_ARMS = ("l188", "l162", "r42q1", "r16q1", "flavf0", "flavf1", "flavf1r", "rand")
JC2_DIFFS = (("flavf0", "l188"), ("flavf1", "l188"), ("flavf1r", "l188"), ("rand", "l188"), ("l162", "l188"),
             ("r42q1", "l162"), ("r16q1", "r42q1"), ("rand", "r16q1"), ("rand", "r42q1"), ("flavf0", "r16q1"),
             ("flavf1", "flavf0"), ("flavf1r", "flavf0"), ("rand", "flavf1"), ("rand", "flavf1r"))
JC2_STEP = ("r16q1", "r42q1", "l162")       # the 43 -> 17 step against the 162 -> 43 step
JC2_FLAGS = "--headline --native-groups"
JC2_OUT = f"{L3_V5}/probe_jc2bal_N1000000.all.json"


def jc2_inputs() -> dict[str, str]:
    return {m: f"{R5.V5['fits_v5_probs']}/{m}/best70/jc2bal_N1000000.npz" for a in JC2_ARMS for m in models(a)}


def jc2_groups(k: int = 3) -> list[list[str]]:
    ms = list(jc2_inputs())
    return [ms[i::k] for i in range(k)]


def jc2_cache(group: list[str], cache: str = CACHE) -> str:
    inp = jc2_inputs()
    return (f"python3 experiments/STATS/paired_bootstrap.py --task npz "
            + " ".join(f"--model {m}={inp[m]}" for m in group) + f" {JC2_FLAGS} --b {B} --cache {cache} --cache-only")


def jc2_file() -> str:
    a, b, c = JC2_STEP
    step = f"--step step-{a}-{b}-{c}={','.join(models(a))}/{','.join(models(b))}/{','.join(models(c))}"
    return pb(JC2_OUT, [f"--model {m}={p}" for m, p in jc2_inputs().items()],
              f"{JC2_FLAGS} {diffs(JC2_DIFFS)} {step} --cache {CACHE}")


# ------------------------------------------------------------------ the other probe rows
PAIR_DIFFS = {"bb_vs_cc": (("r42q1", "l188"), ("r16q1", "l188"), ("flavf1", "r16q1"), ("flavf1r", "r16q1"),
                           ("flavf0", "r16q1")),
              "bc_vs_bq_cs": (("l162", "l188"), ("r42q1", "l188"), ("r16q1", "l188"), ("flavf1", "r16q1"),
                              ("flavf1r", "r16q1"), ("flavf0", "r16q1"), ("flavf0", "l188")),
              "bbqq_vs_ccqq": (("r42q1", "l188"), ("r16q1", "l188"), ("flavf1r", "flavf1")),
              **{p: (("l162", "l188"), ("r42q1", "l188"), ("r16q1", "l188"))
                 for p in ("qcd_bb_vs_cc", "qcd_b_vs_light", "qcd_c_vs_light")},
              "ee_vs_mm": (("r42q1", "l188"), ("r42q1", "initbn"))}
COARSE = "R16_Q1"                       # the 17 merged groups (probe_bal.py --coarse)
COARSE_DIFFS = (("l162", "l188"), ("r42q1", "l188"), ("r16q1", "l188"))
SIZES = (1_000, 10_000, 100_000)        # Fig. A8's linear probes below 1e6
SIZE_DIFFS = (("r42q1", "l188"), ("r16q1", "l188"))


def _arms(pairs) -> list[str]:
    return list(dict.fromkeys(x for p in pairs for x in p))


def pair_path(pair: str, m: str) -> str:
    """The probe's stored test outputs: X->ee vs X->mumu on every test jet of the pair (fits_v5_sib2), the others
    fits_v5's; a BatchNorm-recomputed random network's are init-sK's init_bn."""
    probs = R5.V5["fits_v5_sib2_probs" if pair == "ee_vs_mm" else "fits_v5_probs"]
    run, ck = (m.replace("initbn-", "init-"), "init_bn") if m.startswith("initbn-") else (m, "best70")
    return f"{probs}/{run}/{ck}/jc2bal_pairs_{pair}.npz"


def probe_inputs() -> dict[str, dict[str, str]]:
    """{output file: {model: input}} of every probe file but JetClass-II at 1e6."""
    out = {f"{L3_V5}/probe_jc2bal_pairs_{p}.json": {m: pair_path(p, m) for a in _arms(d) for m in models(a)}
           for p, d in PAIR_DIFFS.items()}
    out[f"{L3_V5}/probe_jc2bal_{COARSE}_N1000000.json"] = {
        m: f"{R5.V5['fits_v5_coarse_probs']}/{m}/best70/jc2bal_{COARSE}_N1000000.npz" for a in _arms(COARSE_DIFFS)
        for m in models(a)}
    for n in SIZES:
        out[f"{L3_V5}/probe_jc2bal_N{n}.v4.json"] = {m: f"{LP}/fits_v4_probs/{m}/best70/jc2bal_N{n}.npz"
                                                     for a in _arms(SIZE_DIFFS) for m in models(a)}
    fill = R5.V5["fits_v5_fill_probs"]
    out[f"{L3_V5}/probe_top_N1200000.fill.json"] = {
        **{m: f"{fill}/{m}/best70.itop4/top_N1200000.npz" for m in models("l188")},
        **{m: f"{fill}/{m}/init.itop4/top_N1200000.npz" for m in models("init")},
        **{m: f"{fill}/{m.replace('initbn-', 'init-')}/init_bn.itop4/top_N1200000.npz" for m in models("initbn")},
        **{f"{m}.zero": f"{fill}/{m}/best70/top_N1200000.npz" for m in models("l188")},
        **{f"{m}.zero": f"{fill}/{m.replace('initbn-', 'init-')}/init_bn/top_N1200000.npz" for m in models("initbn")}}
    out[f"{L3_V5}/probe_qg_N1600000.fill.json"] = {
        **{m: f"{fill}/{m}/best70.nqg/qg_N1600000.npz" for m in models("l188")},
        **{m: f"{fill}/{m}/init.nqg/qg_N1600000.npz" for m in models("init")},
        **{m: f"{fill}/{m.replace('initbn-', 'init-')}/init_bn.nqg/qg_N1600000.npz" for m in models("initbn")}}
    return out


def probe_flags(out: str) -> str:
    name = out.rsplit("/", 1)[1]
    if name.startswith("probe_jc2bal_pairs_"):
        return "--headline " + diffs(PAIR_DIFFS[name.removeprefix("probe_jc2bal_pairs_").removesuffix(".json")])
    if name.startswith(f"probe_jc2bal_{COARSE}"):
        return "--headline " + diffs(COARSE_DIFFS)
    if name.endswith(".v4.json"):
        return "--headline-only " + diffs(SIZE_DIFFS)
    l188, zero = ",".join(models("l188")), ",".join(f"{m}.zero" for m in models("l188"))
    fills = f"--diff 188-init={l188}/{','.join(models('init'))} --diff 188-initbn={l188}/{','.join(models('initbn'))}"
    if name.startswith("probe_top_"):
        fills += (f" --diff 188draws-188zero={l188}/{zero}"
                  f" --diff 188zero-initbnzero={zero}/{','.join(f'{m}.zero' for m in models('initbn'))}")
    return "--headline " + fills


# ------------------------------------------------------------------ fine-tuned on the full sets (ParT's recipe, not rerun)
SCRATCH = [f"scratch-v2-s{k}" for k in (1, 2, 3)]


def ft_inputs() -> dict[str, tuple[str, dict[str, str]]]:
    """{output: (task, {model: cell})}: 188 classes against from scratch on the full top set (R30), and on the full
    quark/gluon set R30 on Herwig over R30 on Pythia per network (--second-task) and Herwig minus Pythia jets."""
    cells = lambda leg, n: {**{m: f"{FT}/best70/{leg}/{m.removeprefix('mtx-')}/N{n}/s1" for m in models("l188")},  # noqa: E731
                            **{m: f"{FT}/ref_v100/bench/{leg}/scratch-v2/N{n}/{m.rsplit('-', 1)[1]}" for m in SCRATCH}}
    return {f"{L3}/ft_top_N1200000.scratch.json": ("top", cells("leg_top", 1_200_000)),
            f"{L3}/ft_qg_N1600000.herwig_over_pythia.json": ("qg", cells("leg_qg", 1_600_000))}


def ft_flags(out: str) -> str:
    d = f"--diff 188-scratch={','.join(models('l188'))}/{','.join(SCRATCH)}"
    if "herwig" in out:
        d += " --second-task qg_herwig" + "".join(f" --diff herwig-pythia-{w}={','.join(f'{m}@second' for m in ms)}/{','.join(ms)}"
                                                  for w, ms in (("188", models("l188")), ("scratch", SCRATCH)))
    return "--headline " + d


def mass() -> list[str]:
    """mass_sd_boot.py with tab:tests' three differences into MASS_V4, once checked to hold MASS_V3 unchanged."""
    new = MASS_V4.replace(".json", ".new.json")
    check = ("import json, sys; o, n = (json.load(open(f)) for f in sys.argv[1:]); "
             "ok = o['groups'] == n['groups'] and all(n['diffs'][k] == v for k, v in o['diffs'].items()); "
             "print('v3 held unchanged in v4:', ok); sys.exit(0 if ok else 42)")
    return [f"if [ ! -e {MASS_V4} ]; then",
            f"  [ -e {new} ] || python3 experiments/EVAL/mass_sd_boot.py --root {R5.ATLAS_V2} --out {new}",
            f"  python3 -c \"{check}\" {MASS_V3} {new}",
            f"  mv -n {new} {MASS_V4}",
            "fi"]


def check_inputs(paths) -> str:
    """Every input must exist before anything runs (exit 42: the index fails, never retried)."""
    return ("N=0; for F in " + " ".join(sorted(set(paths))) + '; do [ -e "${F}" ] || { echo "MISSING ${F}"; N=$((N+1)); }; '
            'done; [ ${N} = 0 ] || { echo "FATAL: ${N} inputs missing"; exit 42; }; echo "all inputs present"')


def rest_inputs() -> list[str]:
    return ([p for d in probe_inputs().values() for p in d.values()]
            + [p for _t, d in ft_inputs().values() for p in d.values()] + [MASS_V3, f"{R5.ATLAS_V2}/DONE"])


def all_inputs() -> list[str]:
    return list(jc2_inputs().values()) + rest_inputs()


def rest() -> list[str]:
    lines = [pb(o, [f"--model {m}={p}" for m, p in d.items()], probe_flags(o)) for o, d in probe_inputs().items()]
    lines += [pb(o, [f"--model {m}={p}" for m, p in d.items()], ft_flags(o), t) for o, (t, d) in ft_inputs().items()]
    return lines + mass()


def wait_cache(hours: int = 30) -> str:
    fs = " ".join(f"{CACHE}/{m}.npz" for m in jc2_inputs())
    return ("T0=$(date +%s)\n"
            f'while M=$(N=0; for F in {fs}; do [ -e "${{F}}" ] || N=$((N+1)); done; echo ${{N}}); [ "${{M}}" -gt 0 ]; do\n'
            f'  [ $(( $(date +%s) - T0 )) -lt {hours * 3600} ] || {{ echo "FATAL: ${{M}} cache files missing"; exit 42; }}\n'
            '  echo "$(date -u +%FT%TZ) waiting for ${M} cache files"; sleep 300\n'
            "done")


def peak(cmd: str) -> str:
    """cmd under a wrapper that prints its peak resident memory and seconds (the job's sizing)."""
    return ("python3 -c 'import resource, subprocess, sys, time; t = time.time(); r = subprocess.call(sys.argv[1:]); "
            "print(\"PEAK_RSS_MIB\", resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss // 1024, \"SECONDS\", "
            "round(time.time() - t)); sys.exit(r)' " + cmd)


def _spec(tag: str, name: str, rows: list[str], parallelism: int, mem: str, what: str) -> str:
    body = f"""set -euo pipefail
{R5.JB._start(tag)}
{R5.threads(CPU)}
{R5.guard()}
mkdir -p {L3} {L3_V5}
{R5.case_block(rows)}echo "BOOT TESTS PART ${{IDX}} COMPLETE"
"""
    text = R5.indexed(name, body, len(rows), parallelism, mem, CPU, what).replace(
        "# GENERATED by scripts/build_refit_v5_jobs.py", "# GENERATED by scripts/build_boot_tests_jobs.py")
    return R5.pin_cpu(text, CPU_MODEL)


def main_spec(tag: str, mem: str) -> str:
    rows = [check_inputs([jc2_inputs()[m] for m in g]) + "\n  " + jc2_cache(g) for g in jc2_groups()]
    rows += ["\n  ".join([check_inputs(rest_inputs())] + rest()), "\n  ".join(wait_cache().splitlines() + [jc2_file()])]
    return _spec(tag, "boot-tests-raunav", rows, PARALLEL, mem,
                 "tab:tests' paired test-jet bootstraps on stored outputs (PI 2026-11-23, SP-12 step 4)")


HERWIG_OUT = f"{L3}/ft_qg_N1600000.herwig_pythia_unpaired.json"


def herwig_spec(tag: str, mem: str) -> str:
    """boot-tests-herwig-raunav (coordinator 2026-11-23): each network's R30 on Herwig minus on Pythia jets, two test
    samples, so unpaired (paired_bootstrap: boot_sd^2 = each side's mean's bootstrap variance, added), into HERWIG_OUT."""
    _t, cells = ft_inputs()[f"{L3}/ft_qg_N1600000.herwig_over_pythia.json"]
    flags = "--headline --second-task qg_herwig" + "".join(
        f" --diff herwig-pythia-{w}={','.join(f'{m}@second' for m in ms)}/{','.join(ms)}"
        for w, ms in (("188", models("l188")), ("scratch", SCRATCH)))
    row = "\n  ".join([check_inputs(cells.values()), pb(HERWIG_OUT, [f"--model {m}={p}" for m, p in cells.items()],
                                                        flags, "qg")])
    return _spec(tag, "boot-tests-herwig-raunav", [row], 1, mem,
                 "tab:tests: R30 on Herwig minus on Pythia jets per network, unpaired (coordinator 2026-11-23)")


def helper_spec(tag: str, mem: str, ms: list[str]) -> str:
    """boot-tests-helper-raunav: the networks ms of the slow cache parts into the same CACHE, in this order (each part
    reads a network's cache file before computing it, so whichever pod writes it first saves the other the work)."""
    inp = jc2_inputs()
    if not ms or set(ms) - set(inp):
        raise SystemExit(f"FATAL: {sorted(set(ms) - set(inp))} are not networks of the JetClass-II file")
    row = "\n  ".join([check_inputs([inp[m] for m in ms])]
                       + [f"[ -e {CACHE}/{m}.npz ] || " + jc2_cache([m]) for m in ms])
    return _spec(tag, "boot-tests-helper-raunav", [row], 1, mem,
                 "tab:tests: JetClass-II networks of the slow cache parts, into the same cache")


# ------------------------------------------------------------------ review of render 5c40d69 (docs/REVIEW_5c40d69_TRIAGE.md, BS)
# B3: the b-quark (and random) flavor boundary - 188 classes on F4-mech's two pairs ("recovers part"); the kinematic-inputs
# network's sentences (results_bench.tex:26, :41) and "both lie below every fine-tuned model" (:36-37); every pretrained
# model's MLP mass resolution against the random network's (results_mass.tex:29-30). A7 and the accuracy rows of B3 are
# in probe_jc2bal_N1000000.all.json already, and F9b's per-bin rows in mass_sd_boot_v4.json.
MASS_V5 = MASS_V3.replace("_v3.json", "_v5.json")
KIN = "mtx-l188kin-s1"
REVIEW_PAIRS = {p: f"{L3_V5}/probe_jc2bal_pairs_{p}.flav188.json" for p in ("bb_vs_cc", "bc_vs_bq_cs")}
REVIEW_KIN_LP = f"{L3_V5}/probe_top_N1200000.kin.json"
REVIEW_FT = f"{L3}/ft_top_N1200000.review.json"


def review_ft_inputs() -> dict[str, str]:
    """Every fine-tuned network on the full top set, the kinematic-inputs network fine-tuned, and the zero-filled linear
    probes of 188 classes and of the BatchNorm-recomputed random network (.npz: read as probes whatever --task says)."""
    fill = R5.V5["fits_v5_fill_probs"]
    out = {m: f"{FT}/best70/leg_top/{m.removeprefix('mtx-')}/N1200000/s1" for a in R5.BOOTALL_ARMS for m in models(a)}
    out[KIN] = f"{FT}/best70_kin/leg_top/{KIN.removeprefix('mtx-')}/N1200000/s1"
    out.update({f"{m}.lpzero": f"{fill}/{m}/best70/top_N1200000.npz" for m in models("l188")})
    out.update({f"{m}.lpzero": f"{fill}/{m.replace('initbn-', 'init-')}/init_bn/top_N1200000.npz" for m in models("initbn")})
    return out


def review_ft_flags() -> str:
    z188, zbn = (",".join(f"{m}.lpzero" for m in models(a)) for a in ("l188", "initbn"))
    d = "".join(f" --diff {a}-188lpzero={','.join(models(a))}/{z188} --diff {a}-initbnlpzero={','.join(models(a))}/{zbn}"
                for a in R5.BOOTALL_ARMS)
    return f"--headline --diff kin-188={KIN}/{','.join(models('l188'))}" + d


def review_a() -> list[str]:
    """The pairs, the kinematic-inputs linear probe, and mass_sd_boot_v5.json (v4 held unchanged)."""
    lines = []
    for p, out in REVIEW_PAIRS.items():
        ms = [m for a in ("flavf1", "flavf1r", "l188") for m in models(a)]
        lines.append(pb(out, [f"--model {m}={pair_path(p, m)}" for m in ms],
                        "--headline " + diffs((("flavf1", "l188"), ("flavf1r", "l188")))))
    draws = {m: f"{R5.V5['fits_v5_fill_probs']}/{m}/best70.itop4/top_N1200000.npz" for m in models("l188")}
    lines.append(pb(REVIEW_KIN_LP, [f"--model {m}={p}" for m, p in draws.items()]
                    + [f"--model {KIN}={R5.V5['fits_v5_probs']}/{KIN}/best70/top_N1200000.npz"],
                    f"--headline --diff 188draws-kin={','.join(draws)}/{KIN}"))
    new = MASS_V5.replace(".json", ".new.json")
    check = ("import json, sys; o, n = (json.load(open(f)) for f in sys.argv[1:]); "
             "ok = o['groups'] == n['groups'] and all(n['diffs'][k] == v for k, v in o['diffs'].items()); "
             "print('v4 held unchanged in v5:', ok); sys.exit(0 if ok else 42)")
    return lines + [f"if [ ! -e {MASS_V5} ]; then",
                    f"  [ -e {new} ] || " + peak(f"python3 experiments/EVAL/mass_sd_boot.py --root {R5.ATLAS_V2} --out {new}"),
                    f"  python3 -c \"{check}\" {MASS_V4} {new}",
                    f"  mv -n {new} {MASS_V5}",
                    "fi"]


def review_inputs() -> list[str]:
    out = [pair_path(p, m) for p in REVIEW_PAIRS for a in ("flavf1", "flavf1r", "l188") for m in models(a)]
    out += [f"{R5.V5['fits_v5_fill_probs']}/{m}/best70.itop4/top_N1200000.npz" for m in models("l188")]
    out += [f"{R5.V5['fits_v5_probs']}/{KIN}/best70/top_N1200000.npz", MASS_V4, f"{R5.ATLAS_V2}/DONE"]
    return out + list(review_ft_inputs().values())


def review_specs(tag: str, mem: str) -> dict[str, str]:
    """boot-review-a-raunav and boot-review-b-raunav (one pod each, launched apart)."""
    a = "\n  ".join([check_inputs(review_inputs())] + review_a())
    b = "\n  ".join([check_inputs(review_ft_inputs().values()),
                     pb(REVIEW_FT, [f"--model {m}={p}" for m, p in review_ft_inputs().items()], review_ft_flags(), "top")])
    what = "tab:tests rows the review of render 5c40d69 found missing (BS)"
    return {"boot-review-a-raunav": _spec(tag, "boot-review-a-raunav", [a], 1, mem, what + ": pairs, kinematic probe, mass"),
            "boot-review-b-raunav": _spec(tag, "boot-review-b-raunav", [b], 1, mem, what + ": fine-tuned top")}


def review_smoke_spec(tag: str, mem: str) -> str:
    """One row first: 188 classes fine-tuned against its own zero-filled linear probe on the full top set (a cell and an
    .npz on the same jets), with its peak memory and seconds."""
    inp = review_ft_inputs()
    ms = models("l188") + [f"{m}.lpzero" for m in models("l188")]
    row = (f"python3 experiments/STATS/paired_bootstrap.py --task top " + " ".join(f"--model {m}={inp[m]}" for m in ms)
           + f" --headline --diff l188-188lpzero={','.join(models('l188'))}/{','.join(ms[3:])} --b {B}"
           + f" --out {SMOKE}/ft_top_review_smoke.json")
    body = "\n  ".join([check_inputs(review_inputs()), f"mkdir -p {SMOKE}", f"[ -e {SMOKE}/ft_top_review_smoke.json ] || "
                        + peak(row)])
    return _spec(tag, "boot-review-smoke-raunav", [body], 1, mem, "review rows, smoke: inputs and one row")


# ------------------------------------------------------------------ second review, render dfe5942 (docs/REVIEW_dfe5942_TRIAGE.md, BS2)
# from the JetClass-II cache (no network recomputed): the b-quark and random flavor boundary - 17 classes (results_jc.tex
# "lie between the 17-class level and the random partitions ... in errors and in accuracy"), and the "costliest single
# merge" steps against the 188 -> 162 step: (43 - 162) - (162 - 188) and (17 - 43) - (162 - 188)
REVIEW2_OUT = f"{L3_V5}/probe_jc2bal_N1000000.review2.json"


def review2_cmd(out: str, arms, diff_pairs, steps) -> str:
    inp = jc2_inputs()
    ms = [m for a in arms for m in models(a)]
    st = "".join(f" --step step-{'-'.join('188' if x == 'l188' else x for x in s)}="
                 + "/".join(",".join(models(x)) for x in s) for s in steps)
    return (f"python3 experiments/STATS/paired_bootstrap.py --task npz " + " ".join(f"--model {m}={inp[m]}" for m in ms)
            + f" {JC2_FLAGS} {diffs(diff_pairs)}{st} --cache {CACHE} --b {B} --out {out}")


REVIEW2 = (("l188", "l162", "r42q1", "r16q1", "flavf1", "flavf1r"), (("flavf1", "r16q1"), ("flavf1r", "r16q1")),
           (("r42q1", "l162", "l188"), ("r16q1", "r42q1", "l162", "l188")))


def review2_spec(tag: str, mem: str, smoke: bool = False) -> str:
    """boot-review2-raunav (or its smoke: one row, with peak memory and seconds) from the JetClass-II cache."""
    if smoke:
        out = f"{SMOKE}/review2_smoke.json"
        body = f"mkdir -p {SMOKE}\n  [ -e {out} ] || " + peak(review2_cmd(out, ("flavf1", "r16q1"), (("flavf1", "r16q1"),), ()))
        return _spec(tag, "boot-review2-smoke-raunav", [body], 1, mem, "second review rows, smoke: one row from the cache")
    body = f"[ -e {REVIEW2_OUT} ] || " + peak(review2_cmd(REVIEW2_OUT, *REVIEW2))
    return _spec(tag, "boot-review2-raunav", [body], 1, mem, "second review rows (BS2) from the JetClass-II cache")


def smoke_spec(tag: str, mem: str) -> str:
    inp = {m: pair_path("bb_vs_cc", m) for a in ("r42q1", "l188") for m in models(a)}
    row = pb(f"{SMOKE}/probe_jc2bal_pairs_bb_vs_cc.json", [f"--model {m}={p}" for m, p in inp.items()],
             "--headline " + diffs((("r42q1", "l188"),)))
    one = jc2_cache(["mtx-l188-s1"], f"{SMOKE}/cache")
    body = "\n  ".join([check_inputs(all_inputs()), f"mkdir -p {SMOKE}", peak(row.split(" || ", 1)[1]),
                        peak(one), f"ls -la {SMOKE} {SMOKE}/cache"])
    return _spec(tag, "boot-tests-smoke-raunav", [body], 1, mem,
                 "tab:tests' bootstraps, smoke: inputs, one row, one JetClass-II network's time and memory")


# ------------------------------------------------------------------ the fine-tuned rows after the reruns (2026-11-24)
# Part A of the fixed-schedule reruns is done: the paired test-jet bootstrap of every fine-tuned difference on the one
# rule's cells at each row's recorded rate (build_ft_budget_jobs.paper_fixed_cells), one new file a task and size in
# L3_FIXED, every difference of the task in it under both names the readers use (BL.DIFFS, R5.BOOTALL_DIFFS). Its own
# folder: the pull keeps bootstrap/<family>/ as it is, and BL.OUT_RERUN's files go to l3_rerun/ outside bootstrap/.
# The partitions' cells below 1e6 are part B (results book only), so those files hold the label sets, the mass
# networks and from scratch. Top and quark/gluon on the full sets keep ParT's recipe: their files stay in L3.
L3_FIXED = "/data/results/bootstrap/l3_fixed"
FTFIX_SETS = ("l188", "l162", "r42q1", "r16q1", "l162mass", "r16q1mass")
FTFIX_PARTS = ("flavf0", "flavf1", "flavf1r", "rand")
# (paired_bootstrap's task, the builder's task, the file's name, size, with the partitions)
FTFIX_FILES = (("jc2b", "jc2b", "jc2bal", 1_000_000, True), ("jc1_all", "jc1", "jc1", 1_000_000, True),
               ("jc1_all", "jc1", "jc1", 100_000, False), ("jc2b", "jc2b", "jc2bal", 100_000, False),
               ("jc2b", "jc2b", "jc2bal", 10_000, False))
FTFIX_BENCH = tuple((t, "top" if t == "top" else "qg", t, n, False) for n in SIZES for t in ("top", "qg", "qg_herwig"))
FTFIX_SMOKE = ("mtx-l188-s1", "mtx-l188-s2")


def ftfix_out(name: str, n: int) -> str:
    return f"{L3_FIXED}/ft_{name}_N{n}.json"


def ftfix_cells(task: str, n: int, parts: bool) -> dict[str, str]:
    """{model: cell} of a task and size at its row's recorded rate: the label sets and mass networks, the partitions
    with parts, and the three from-scratch trainings; a network without its cell is an error, not a gap."""
    fb = _load("build_ft_budget_jobs", "scripts/build_ft_budget_jobs.py")
    cells = {(c["init"], c["s"]): c["out"] for c in fb.paper_fixed_cells() if c["task"] == task and c["n"] == n}
    want = [(m, (m.removeprefix("mtx-"), 1)) for a in FTFIX_SETS + (FTFIX_PARTS if parts else ()) for m in models(a)]
    want += [(m, ("", int(m.rsplit("-s", 1)[1]))) for m in SCRATCH]
    if any(k not in cells for _, k in want):
        raise SystemExit(f"FATAL: no {task} cell at {n} for {[m for m, k in want if k not in cells]}")
    return {m: cells[k] for m, k in want}


def ftfix_diffs(parts: bool) -> str:
    side = lambda a: ",".join(SCRATCH if a == "scratch" else models(a))  # noqa: E731
    arms = FTFIX_SETS + (FTFIX_PARTS if parts else ())
    named = {**BL.DIFFS, **{k: v for k, v in R5.BOOTALL_DIFFS.items() if v[0] in arms},
             "r42q1-l162": ("r42q1", "l162"), "r16q1-r42q1": ("r16q1", "r42q1"), "188-scratch": ("l188", "scratch"),
             **({dname(a, "r16q1"): (a, "r16q1") for a in FTFIX_PARTS} if parts else {})}
    return " ".join(f"--diff {k}={side(a)}/{side(b)}" for k, (a, b) in named.items())


def ftfix_row(files) -> str:
    lines, done = [], []
    for pb_task, task, name, n, parts in files:
        cells = ftfix_cells(task, n, parts)
        done += [f"{c}/DONE" for c in cells.values()]
        cmd = pb(ftfix_out(name, n), [f"--model {m}={c}" for m, c in cells.items()], "--headline " + ftfix_diffs(parts), pb_task)
        lines.append(" || ".join((lambda a, b: (a, peak(b)))(*cmd.split(" || ", 1))))
    return "\n  ".join([check_inputs(done), f"mkdir -p {L3_FIXED}"] + lines)


def ftfix_spec(tag: str, mem: str, smoke: bool = False) -> str:
    """boot-ftfix-raunav: one pod a JetClass-II or JetClass file and one for the nine top and quark/gluon files below
    the full sets; its smoke (boot-ftfix-smoke-raunav): two networks of each 1e6 file into SMOKE, for peak and seconds."""
    if smoke:
        rows = []
        for pb_task, task, name, n, _ in FTFIX_FILES[:2]:
            cells = {m: c for m, c in ftfix_cells(task, n, False).items() if m in FTFIX_SMOKE}
            out = f"{SMOKE}/ftfix_{name}_N{n}.json"
            cmd = pb(out, [f"--model {m}={c}" for m, c in cells.items()], "--headline", pb_task).split(" || ", 1)[1]
            rows.append("\n  ".join([check_inputs(f"{c}/DONE" for c in cells.values()), f"mkdir -p {SMOKE}",
                                     f"[ -e {out} ] || " + peak(cmd)]))
        return _spec(tag, "boot-ftfix-smoke-raunav", rows, 2, mem,
                     "fine-tuned rows after the reruns, smoke: two networks of each 1e6 file, peak memory and seconds")
    rows = [ftfix_row([f]) for f in FTFIX_FILES] + [ftfix_row(FTFIX_BENCH)]
    return _spec(tag, "boot-ftfix-raunav", rows, len(rows), mem,
                 "paired test-jet bootstraps of the fine-tuned rows on the fixed-schedule rerun cells (2026-11-24)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", required=True, help="the pushed tag the pods clone")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--herwig", action="store_true", help="boot-tests-herwig-raunav, the unpaired Herwig - Pythia rows")
    ap.add_argument("--helper", nargs="+", metavar="MODEL", help="boot-tests-helper-raunav: these networks into the cache")
    ap.add_argument("--node", help="one node of CPU_MODEL's (R5.CPU_NODES) instead of all of them")
    ap.add_argument("--review", choices=("smoke", "a", "b"), help="the review rows: the smoke, or job a or b")
    ap.add_argument("--review2", choices=("smoke", "job"), help="the second review's rows from the cache")
    ap.add_argument("--ftfix", choices=("smoke", "job"), help="the fine-tuned rows on the fixed-schedule rerun cells")
    ap.add_argument("--mem", default="16Gi", help="memory request and limit of every pod")
    a = ap.parse_args(argv)
    if not re.fullmatch(r"mtx-s\d+\.\d+", a.tag):
        raise SystemExit(f"FATAL: --tag {a.tag} is not an mtx-sN.NN tag")
    review = ({"smoke": ("boot-review-smoke-raunav", review_smoke_spec(a.tag, a.mem))}
              | {k[12]: (k, v) for k, v in review_specs(a.tag, a.mem).items()}) if a.review else {}
    name, text = (("boot-review2-smoke-raunav" if a.review2 == "smoke" else "boot-review2-raunav",
                   review2_spec(a.tag, a.mem, a.review2 == "smoke")) if a.review2
                  else ("boot-ftfix-smoke-raunav" if a.ftfix == "smoke" else "boot-ftfix-raunav",
                        ftfix_spec(a.tag, a.mem, a.ftfix == "smoke")) if a.ftfix
                  else review[a.review] if a.review
                  else ("boot-tests-smoke-raunav", smoke_spec(a.tag, a.mem)) if a.smoke
                  else ("boot-tests-herwig-raunav", herwig_spec(a.tag, a.mem)) if a.herwig
                  else ("boot-tests-helper-raunav", helper_spec(a.tag, a.mem, a.helper)) if a.helper
                  else ("boot-tests-raunav", main_spec(a.tag, a.mem)))
    if a.node:
        nodes = R5.CPU_NODES[CPU_MODEL]
        old = "values: [" + ", ".join(f'"{n}"' for n in nodes) + "]"
        if a.node not in nodes or text.count(old) != 1:
            raise SystemExit(f"FATAL: {a.node} is not one of the {CPU_MODEL} nodes")
        text = text.replace(old, f'values: ["{a.node}"]')
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / f"job-{name}.yaml"
    p.write_text(text)
    print(p.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
