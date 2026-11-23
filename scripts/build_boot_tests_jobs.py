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


def smoke_spec(tag: str, mem: str) -> str:
    inp = {m: pair_path("bb_vs_cc", m) for a in ("r42q1", "l188") for m in models(a)}
    row = pb(f"{SMOKE}/probe_jc2bal_pairs_bb_vs_cc.json", [f"--model {m}={p}" for m, p in inp.items()],
             "--headline " + diffs((("r42q1", "l188"),)))
    one = jc2_cache(["mtx-l188-s1"], f"{SMOKE}/cache")
    body = "\n  ".join([check_inputs(all_inputs()), f"mkdir -p {SMOKE}", peak(row.split(" || ", 1)[1]),
                        peak(one), f"ls -la {SMOKE} {SMOKE}/cache"])
    return _spec(tag, "boot-tests-smoke-raunav", [body], 1, mem,
                 "tab:tests' bootstraps, smoke: inputs, one row, one JetClass-II network's time and memory")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", required=True, help="the pushed tag the pods clone")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--herwig", action="store_true", help="boot-tests-herwig-raunav, the unpaired Herwig - Pythia rows")
    ap.add_argument("--mem", default="16Gi", help="memory request and limit of every pod")
    a = ap.parse_args(argv)
    if not re.fullmatch(r"mtx-s\d+\.\d+", a.tag):
        raise SystemExit(f"FATAL: --tag {a.tag} is not an mtx-sN.NN tag")
    name, text = (("boot-tests-smoke-raunav", smoke_spec(a.tag, a.mem)) if a.smoke
                  else ("boot-tests-herwig-raunav", herwig_spec(a.tag, a.mem)) if a.herwig
                  else ("boot-tests-raunav", main_spec(a.tag, a.mem)))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / f"job-{name}.yaml"
    p.write_text(text)
    print(p.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
