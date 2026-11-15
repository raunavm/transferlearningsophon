#!/usr/bin/env python3
"""The v5 refits of the frozen read-outs (PI 2026-11-14, after the frozen-read-out audit). CPU jobs, except the
released Sophon checkpoint's fill extraction (sophonx, one GPU pod). Every output goes to a NEW folder, so the
fits_v4 / atlas_res / xbx results stay as they are; every job name holds "raunav"; the in-pod storage guard is
storage_line.FINISHING. Specs in experiments/EVAL/k8s/v2/refit_v5/; plan and launch order in the PI's brief.

  fit      probe_bal.py --grid v5, the cells of fits_v4 (balanced JetClass-II at every size, the pairs, the
           input variants, JetClass, top, quark/gluon with Herwig, the 188-class runs' hidden layer) for the
           27 paper runs and the three untrained trunks, and the kinematic-only run's top probe
           -> v2_linprobe_bal/fits_v5, fits_v5_probs
  fill     probe_bal.py --fill --grid v5 (zero / mean / four draws side by side) -> fits_v5_fill, fits_v5_fill_probs
  sib      probe_bal.py --sibling-pairs --sibling-test --grid v5 (X->ee vs X->mumu) -> fits_v5_sib2(_probs)
  coarse   probe_bal.py --coarse --grid v5 -> fits_v5_coarse(_probs)
  qcdhf    qcd_hf_rejection.py on fits_v5_probs -> fits_v5_qcdhf
  nat      linear_probe_v2.py --grid v5 --datasets jc2_188: the natural-composition JetClass-II probe and its
           pairs at best70 and best70_bn (tab:bn, tab:balance, MlstBnMaxDev*) -> v2_linprobe/fits_v5_nat
  sophonx  GPU, one pod: the released Sophon checkpoint's features on the top and quark/gluon splits with the
           pretrained rows' fills (mean, four draws; zero re-extracted) -> v2_linprobe_sophon_fill
  sophon   probe_released.py --grid v5 (the fits_v4 cells, the coarse sets) and --fill-root (the fills), and
           qcd_hf_rejection.py -> v2_linprobe_sophon/fits_v5, fits_v5_fill, fits_v5_qcdhf
  xbx      fit_xbx.py --linear-only --grid v5, X->bb and X->bs, every model -> xbx/fits_ihep_v5/<signal>
  mlp      the class-balanced JetClass-II MLP probe and its pairs (probe_bal.py --mlp --mlp-datasets jc2bal) for
           every network, the 9 fits_v4_mlp lacks (mass-output runs, random partitions) among them, and the
           kinematic-only run's top MLP; one CPU model (CPU_PIN) -> fits_v5_mlp. The whole row: the 18 ran at
           mtx-s2.34, and a spec cloning that tag is refused by tests/test_spec_pins.py (probe_bal.py changed since)
  atlas    atlas_resolution.py --grid v2 --mlp-max-epochs MASS_MLP_EPOCHS on one CPU model (CPU_PIN), every
           model of the mass table -> atlas_res_v2; rebin: atlas_rebin.py -> atlas_res_fine_v2
  boot     paired_bootstrap.py on the v5 probes: balanced JetClass-II at 1e6 and its pairs, JetClass at 1e6
           (fits_v5_probs), top and quark/gluon (Herwig) at every size under the chosen substitutions
           (fits_v5_fill_probs, FILL_TAG) -> bootstrap/probe_v5
  bootl3   decision L3's probe parts (build_boot_l3_jobs: probe-jc, probe-bin) on the v5 probes -> bootstrap/l3_v5
  checks   refit_checks.py on every family above once all exist: grid edges, MLP best epochs in the last tenth,
           one CPU model for the mass MLP, every model present; the job fails if any rule fails

    python3 scripts/build_refit_v5_jobs.py --tag mtx-s2.98 all
"""
from __future__ import annotations

import argparse
import importlib.util
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import storage_line  # noqa: E402

OUT_DIR = ROOT / "experiments" / "EVAL" / "k8s" / "v2" / "refit_v5"
LINE = storage_line.check(storage_line.FINISHING)


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


JB = _load("build_jc2bal_jobs", "scripts/build_jc2bal_jobs.py")
LP = JB.LP_BAL                                          # /data/results/eval/v2_linprobe_bal
NAT_ROOT = "/data/results/eval/v2_linprobe"             # natural-composition features (build_linprobe_jobs LP_ROOT)
V5 = {k: f"{LP}/{k}" for k in ("fits_v5", "fits_v5_probs", "fits_v5_fill", "fits_v5_fill_probs", "fits_v5_sib2",
                                "fits_v5_sib2_probs", "fits_v5_coarse", "fits_v5_coarse_probs", "fits_v5_qcdhf")}
NAT = f"{NAT_ROOT}/fits_v5_nat"
SOPHON_FILL_ROOT = "/data/results/eval/v2_linprobe_sophon_fill"
SOPHON_V5 = {k: f"{JB.SOPHON_ROOT}/{k}" for k in ("fits_v5", "fits_v5_probs", "fits_v5_fill", "fits_v5_fill_probs",
                                                 "fits_v5_qcdhf")}
XBX_ROOT, XBX_V5 = "/data/results/eval/xbx", "/data/results/eval/xbx/fits_ihep_v5"
XBX_SIGNALS = ("xbb", "xbs")
MLP_V5 = f"{LP}/fits_v5_mlp"
ATLAS_V2, ATLAS_FINE_V2 = "/data/results/eval/atlas_res_v2", "/data/results/eval/atlas_res_fine_v2"
# the mass MLP's cap: 60 at atlas_res; best epochs were not recorded, and probe.py's classification MLP left 93 of 360 fits
# still improving under the same 60 / 8 rule, so the refit starts at 200 (patience and gain unchanged); refit_checks
# fails the family if any best epoch lies in the last tenth, and then every network is refitted at a longer cap
MASS_MLP_EPOCHS = 200
# one CPU model for the mass MLP (latent_scale_probe.py: the CPU model is part of the result): AMD EPYC Rome
# (family 23, model 49), the most numerous CPU-only nodes in us-west on 2026-11-14 (32), and the model on which the
# committed mass MLP values were reproduced bit for bit (latent_scale_probe.py, v1err-mass-pinned-check)
CPU_PIN = {"feature.node.kubernetes.io/cpu-model.vendor_id": "AMD", "feature.node.kubernetes.io/cpu-model.family": "23",
           "feature.node.kubernetes.io/cpu-model.id": "49"}
BOOT_V5, L3_V5 = "/data/results/bootstrap/probe_v5", "/data/results/bootstrap/l3_v5"
CHECKS_ROOT = "/data/results/eval/refit_v5_checks"     # outside the families, so a pull of a family takes no summary
# the substitutions chosen on validation (mlst_results.chosen_substitutions on fits_v4_fill, 2026-11-14: top draws,
# quark/gluon mean); the reader refuses probe_v5 if the v5 choice differs (mlst_results.check_boot_fills)
FILL_TAG = {"top": "itop4", "qg": "nqg", "qg_herwig": "nqg"}
KIN = JB.KIN_MODELS[0][0]                              # mtx-l188kin-s1 (top only)


def guard() -> str:
    return ('USED=$(df --output=pcent /data | tail -1 | tr -dc 0-9)\n'
            f'[ "${{USED}}" -le {LINE} ] || {{ echo "FATAL: /data at ${{USED}}%"; exit 42; }}')


def threads(n: str) -> str:
    return f"export PYTHONUNBUFFERED=1 OMP_NUM_THREADS={n} OPENBLAS_NUM_THREADS={n} MKL_NUM_THREADS={n}"


def indexed(name: str, body: str, n: int, parallelism: int, mem: str, cpu: str, what: str, gpu: bool = False) -> str:
    """build_jc2bal_jobs._indexed (as the fits_v4 jobs), kept off the bad nodes, two retries per index."""
    text = JB._indexed(name, body, n, parallelism, gpu, mem, cpu, what + (". GPU" if gpu else ". CPU"), exclude=True)
    assert text.count("backoffLimitPerIndex: 6\n") == 1
    return text.replace("backoffLimitPerIndex: 6\n", "backoffLimitPerIndex: 2\n").replace(
        "# GENERATED by scripts/build_jc2bal_jobs.py", "# GENERATED by scripts/build_refit_v5_jobs.py")


def single(name: str, body: str, mem: str, cpu: str, what: str, gpu: bool = False) -> str:
    """One pod (an Indexed Job of one index)."""
    return indexed(name, body, 1, 1, mem, cpu, what, gpu)


def cases(rows) -> str:
    return "".join(f"  {i}) {r};;\n" for i, r in enumerate(rows))


def case_block(rows) -> str:
    return ("IDX=${JOB_COMPLETION_INDEX:?not a pod of an Indexed Job}\ncase ${IDX} in\n" + cases(rows)
            + '  *) echo "FATAL: no model for index ${IDX}"; exit 42;;\nesac\n')


def models():
    """(model, checkpoints, variants) of the 27 paper runs and the three untrained trunks (build_jc2bal_jobs.models)."""
    return [(m, c, v) for m, _r, _g, _k, _e, c, v in JB.models()]


def model_names() -> list[str]:
    return [m for m, _c, _v in models()]


def fit_spec(tag: str, parallelism: int) -> str:
    rows = []
    for m, c, v in models():
        extra = f"--variants {' '.join(v)}" + (" --hidden" if m.startswith(JB.HIDDEN) else "")
        rows.append(f'MODEL={m}; CKPTS="{" ".join(c)}"; EXTRA="{extra}"; NEED={LP}/{m}/DONE')
    rows.append(f'MODEL={KIN}; CKPTS="best70"; EXTRA="--old-only top"; NEED={JB.LP_OLD}/{KIN}/top/test/best70/manifest.json')
    body = f"""set -euo pipefail
{JB._start(tag)}
{threads("8")}
{case_block(rows)}[ -f {V5["fits_v5"]}/${{MODEL}}.json ] && {{ echo "${{MODEL}} done"; exit 0; }}
[ -f ${{NEED}} ] || {{ echo "FATAL: ${{MODEL}}'s features are missing (${{NEED}})"; exit 42; }}
{guard()}
python3 experiments/EVAL/probe_bal.py --model ${{MODEL}} --checkpoints ${{CKPTS}} ${{EXTRA}} --grid v5 \\
  --out {V5["fits_v5"]} --probs-out {V5["fits_v5_probs"]}
echo "REFIT5 FIT ${{MODEL}} COMPLETE"
"""
    return indexed("refit5-fit-raunav", body, len(rows), parallelism, "48Gi", "8",
                   "v5 linear probes, fits_v4's cells on the v5 grid (PI 2026-11-14)")


def _per_model(tag: str, name: str, flags: str, out: str, probs: str, need: str, what: str, parallelism: int,
               mem: str, cpu: str) -> str:
    rows = [f'MODEL={m}; CKPTS="{" ".join(c)}"' for m, c, _v in models()]
    body = f"""set -euo pipefail
{JB._start(tag)}
{threads(cpu)}
{case_block(rows)}[ -f {out}/${{MODEL}}.json ] && {{ echo "${{MODEL}} done"; exit 0; }}
[ -f {need} ] || {{ echo "FATAL: ${{MODEL}}'s features are missing ({need})"; exit 42; }}
{guard()}
python3 experiments/EVAL/probe_bal.py --model ${{MODEL}} --checkpoints ${{CKPTS}} {flags} --grid v5 \\
  --out {out} --probs-out {probs}
echo "REFIT5 {name.upper()} ${{MODEL}} COMPLETE"
"""
    return indexed(f"refit5-{name}-raunav", body, len(rows), parallelism, mem, cpu, what)


def fill_spec(tag: str, parallelism: int) -> str:
    return _per_model(tag, "fill", "--fill", V5["fits_v5_fill"], V5["fits_v5_fill_probs"], f"{JB.FILL_ROOT}/${{MODEL}}/DONE",
                      "v5 linear probes on the fills of the absent inputs (PI 2026-11-14)", parallelism, "48Gi", "8")


def sib_spec(tag: str, parallelism: int) -> str:
    return _per_model(tag, "sib", "--sibling-pairs --sibling-test", V5["fits_v5_sib2"], V5["fits_v5_sib2_probs"],
                      f"{JB.LP_BAL}/fits_v4_sib2/${{MODEL}}.json",
                      "v5 X->ee vs X->mumu probe on every test jet of the pair (PI 2026-11-14)", parallelism, "16Gi", "4")


def coarse_spec(tag: str, parallelism: int) -> str:
    return _per_model(tag, "coarse", f"--coarse {' '.join(JB.COARSE)}", V5["fits_v5_coarse"], V5["fits_v5_coarse_probs"],
                      f"{LP}/${{MODEL}}/DONE", "v5 linear probes on the coarse label sets (PI 2026-11-14)",
                      parallelism, "48Gi", "8")


def _wait(paths: list[str], hours: int = 144) -> str:
    """Poll every 30 min until every path exists; give up after `hours` with exit 42."""
    return ("missing () { local n=0 f; for f in " + " ".join(paths) + "; do [ -e \"${f}\" ] || n=$((n+1)); done; echo ${n}; }\n"
            "T0=$(date +%s)\n"
            'while M=$(missing); [ "${M}" -gt 0 ]; do\n'
            f'  [ $(( $(date +%s) - T0 )) -lt {hours * 3600} ] || {{ echo "FATAL: ${{M}} inputs missing"; exit 42; }}\n'
            '  echo "$(date -u +%FT%TZ) waiting for ${M} inputs"; sleep 1800\n'
            "done\n")


def qcdhf_spec(tag: str) -> str:
    loop = "".join(f"[ -f {V5['fits_v5_qcdhf']}/{m}.json ] || python3 experiments/EVAL/qcd_hf_rejection.py --model {m} "
                   f"--checkpoints {' '.join(c)} --probs-root {V5['fits_v5_probs']} --features-root {LP} "
                   f"--out {V5['fits_v5_qcdhf']}\n" for m, c, _v in models())
    body = f"""set -euo pipefail
{JB._start(tag)}
{threads("4")}
{_wait([f"{V5['fits_v5']}/{m}.json" for m in model_names()])}{guard()}
{loop}echo "REFIT5 QCDHF COMPLETE"
"""
    return single("refit5-qcdhf-raunav", body, "24Gi", "4", "QCD heavy-flavor rejection from the v5 probes")


def nat_spec(tag: str, parallelism: int) -> str:
    rows = [f'MODEL={m}; CKPTS="{"init init_bn" if m.startswith("init-") else "best70 best70_bn"}"' for m in model_names()]
    body = f"""set -euo pipefail
{JB._start(tag)}
{threads("8")}
{case_block(rows)}[ -f {NAT}/${{MODEL}}.json ] && {{ echo "${{MODEL}} done"; exit 0; }}
for C in ${{CKPTS}}; do [ -f {NAT_ROOT}/${{MODEL}}/jc2/test/${{C}}/manifest.json ] || {{ echo "FATAL: no ${{C}} features"; exit 42; }}; done
{guard()}
python3 experiments/EVAL/linear_probe_v2.py --root {NAT_ROOT} --model ${{MODEL}} --checkpoints ${{CKPTS}} \\
  --datasets jc2_188 --grid v5 --out {NAT}
echo "REFIT5 NAT ${{MODEL}} COMPLETE"
"""
    return indexed("refit5-nat-raunav", body, len(rows), parallelism, "48Gi", "8",
                   "v5 natural-composition JetClass-II probes, best70 and best70_bn (PI 2026-11-14)")


def sophonx_spec(tag: str) -> str:
    """As build_jc2bal_jobs.sophon_spec's extraction, on the top and quark/gluon splits only, with
    xfill_spec's fills (--variants, the version-2 imputation reference): the zero fill is re-extracted beside them."""
    pb = JB._probe_builder()
    ck = f"/workspace/sophon/{JB.SOPHON_TAG}.pt"
    calls = ""
    for ds in ("top", "qg"):
        sizes, cfg, extra = pb.DATASETS[ds]
        calls += "".join(f'extract {ds} train_N{n} {cfg} "{JB.FILL_VARIANTS[ds]}" {pb.TRAIN_DIR[ds]}/train_N{n}_s1.parquet\n'
                         for n in sizes)
        for split, (maxj, files) in extra.items():
            if maxj:
                raise SystemExit(f"FATAL: {ds}/{split} is read with --max-jets {maxj}")
            calls += f'extract {ds} {split} {cfg} "{JB.FILL_VARIANTS[ds]}" {files}\n'
    out = f"{SOPHON_FILL_ROOT}/{JB.SOPHON_MODEL}"
    body = f"""set -euo pipefail
{JB._start(tag)}
[ -f {out}/DONE ] && {{ echo "{JB.SOPHON_MODEL} fills done"; exit 0; }}
[ -f {JB.REF2} ] || {{ echo "FATAL: {JB.REF2} missing"; exit 42; }}
{guard()}
{JB.GPU_CHECK}
mkdir -p /workspace/sophon
retry python3 -c "import urllib.request; urllib.request.urlretrieve('{JB.SOPHON_URL}', '{ck}')" || exit 1
GOT=$(sha256sum {ck} | cut -d' ' -f1)
[ "${{GOT}}" = "{JB.SOPHON_SHA256}" ] || {{ echo "FATAL: {ck} sha256 ${{GOT}}"; exit 42; }}
# extract DATASET SPLIT CONFIG "VARIANTS" FILES...: one pass, the zero fill and its fills; a split whose every
# directory has its manifest (written last) is done
extract () {{
  OUT={out}/$1/$2; CFGX=$3; VARS=$4; shift 4
  ALL=1; for T in {JB.SOPHON_TAG} $(for V in ${{VARS}}; do echo {JB.SOPHON_TAG}.${{V}}; done); do
    [ -f ${{OUT}}/${{T}}/manifest.json ] || ALL=0; done
  [ ${{ALL}} = 0 ] || {{ echo "${{OUT}} done"; return 0; }}
  python3 experiments/EVAL/extract_v2.py --run-dir /workspace/sophon --rung L188 --num-classes 188 \\
    --num-reg 0 --checkpoints {ck} --data-config ${{CFGX}} --max-jets 0 \\
    --observers --feature-classes --prefix-features 2000000000 --head-prefix 0 --no-anomaly-rows \\
    --diag-stride 0 --no-pooled --data-test "$@" --variants ${{VARS}} --impute-ref {JB.REF2} --out ${{OUT}} || exit 1
}}
{calls}touch {out}/DONE
echo "REFIT5 SOPHONX COMPLETE"
"""
    return single("refit5-sophonx-raunav", body, pb.X_MEM, "4",
                  "the released Sophon checkpoint's features with the pretrained rows' fills (PI 2026-11-14)", gpu=True)


def sophon_spec(tag: str) -> str:
    m, ck = JB.SOPHON_MODEL, JB.SOPHON_TAG
    body = f"""set -euo pipefail
{JB._start(tag)}
{threads("8")}
{_wait([f"{SOPHON_FILL_ROOT}/{m}/DONE"])}{guard()}
[ -f {SOPHON_V5["fits_v5"]}/{m}.json ] || python3 experiments/EVAL/probe_released.py --model {m} --checkpoints {ck} \\
  --root {JB.SOPHON_ROOT} --coarse {" ".join(JB.COARSE)} --grid v5 --out {SOPHON_V5["fits_v5"]} --probs-out {SOPHON_V5["fits_v5_probs"]}
[ -f {SOPHON_V5["fits_v5_fill"]}/{m}.json ] || python3 experiments/EVAL/probe_released.py --model {m} --checkpoints {ck} \\
  --root {JB.SOPHON_ROOT} --fill-root {SOPHON_FILL_ROOT} --grid v5 --out {SOPHON_V5["fits_v5_fill"]} \\
  --probs-out {SOPHON_V5["fits_v5_fill_probs"]}
[ -f {SOPHON_V5["fits_v5_qcdhf"]}/{m}.json ] || python3 experiments/EVAL/qcd_hf_rejection.py --model {m} --checkpoints {ck} \\
  --probs-root {SOPHON_V5["fits_v5_probs"]} --features-root {JB.SOPHON_ROOT} --out {SOPHON_V5["fits_v5_qcdhf"]}
echo "REFIT5 SOPHON COMPLETE"
"""
    return single("refit5-sophon-raunav", body, "48Gi", "8", "v5 probes of the released Sophon checkpoint, with fills")


def xbx_models() -> tuple[list[tuple[str, str]], str, float]:
    """([(model, checkpoint)], the window argument, mX) of build_xbx_jobs' real-sample fits."""
    xb = _load("build_xbx_jobs", "scripts/build_xbx_jobs.py")
    return [(m, ck) for m, _r, _a, _k, _g, ck in xb.models()], xb.window_arg(xb.MX), xb.MX


def xbx_spec(tag: str, parallelism: int) -> str:
    ms, window, mx = xbx_models()
    rows = [f"MODEL={m}; CKPT={ck}" for m, ck in ms]
    fits = "".join(f"[ -f {XBX_V5}/{s}/${{MODEL}}.json ] || python3 experiments/XBS/fit_xbx.py model --root {XBX_ROOT} "
                   f"--model ${{MODEL}} --signals {s} --checkpoint ${{CKPT}} {window} --mx {mx:g} --linear-only --grid v5 "
                   f"--out {XBX_V5}/{s}\n" for s in XBX_SIGNALS)
    body = f"""set -euo pipefail
{JB._start(tag)}
{threads("2")}
{case_block(rows)}for S in {" ".join(XBX_SIGNALS)}; do
  [ -f /data/results/eval/xbx/fits_ihep/${{S}}/${{MODEL}}.json ] || {{ echo "FATAL: ${{MODEL}} has no ${{S}} fit to refit"; exit 42; }}
done
{guard()}
{fits}echo "REFIT5 XBX ${{MODEL}} COMPLETE"
"""
    return indexed("refit5-xbx-raunav", body, len(rows), parallelism, "24Gi", "2",
                   "v5 X->bb / X->bs linear probes, the linear probe only (PI 2026-11-14)")


def mlp_spec(tag: str, parallelism: int) -> str:
    """build_jc2bal_jobs.fit_spec(mlp=True)'s fits (and kin_spec's top MLP) for every network, pinned to CPU_PIN."""
    rows = [f'MODEL={m}; CKPTS="{" ".join(c)}"; DS=jc2bal; NEED={LP}/{m}/DONE' for m, c, _v in models()]
    rows.append(f'MODEL={KIN}; CKPTS="best70"; DS=top; NEED={JB.LP_OLD}/{KIN}/top/test/best70/manifest.json')
    body = f"""set -euo pipefail
{JB._start(tag)}
{case_block(rows)}[ -f {MLP_V5}/${{MODEL}}.json ] && {{ echo "${{MODEL}} done"; exit 0; }}
[ -f ${{NEED}} ] || {{ echo "FATAL: ${{MODEL}}'s features are missing (${{NEED}})"; exit 42; }}
{guard()}
python3 experiments/EVAL/probe_bal.py --model ${{MODEL}} --checkpoints ${{CKPTS}} --mlp --mlp-datasets ${{DS}} --out {MLP_V5}
echo "REFIT5 MLP ${{MODEL}} COMPLETE"
"""
    # fits_v4_mlp's job: CPU 8, 64Gi, no thread exports (probe.py pins the MLP's threads itself)
    return pin_cpu(indexed("refit5-mlp-raunav", body, len(rows), parallelism, "64Gi", "8",
                           "class-balanced JetClass-II MLP probes of every network on one CPU model (PI 2026-11-14)"))


def pin_cpu(text: str) -> str:
    """CPU_PIN added to a spec's required node affinity."""
    old = '                values: ["us-west"]\n'
    if text.count(old) != 1:
        raise SystemExit("FATAL: the spec's region term changed")
    return text.replace(old, old + "".join(f"              - key: {k}\n                operator: In\n                values: [\"{v}\"]\n"
                                           for k, v in CPU_PIN.items()))


def _atlas_module():
    return _load("build_atlas_res_job", "scripts/build_atlas_res_job.py")


def atlas_spec(tag: str, parallelism: int) -> str:
    """build_atlas_res_job.spec into ATLAS_V2 with --grid v2 and the longer MLP cap, pinned to CPU_PIN."""
    ba = _atlas_module()
    text = ba.spec(tag, parallelism)
    subs = [(f"name: {ba.NAME}\n", "name: refit5-atlas-raunav\n", 1),
            (f"OUT={ba.OUT_ROOT}\n", f"OUT={ATLAS_V2}\n", 1),
            ("--mass-v3 ${MASSV3} --out-dir ${OUT} || halt",
             f"--mass-v3 ${{MASSV3}} --out-dir ${{OUT}} --grid v2 --mlp-max-epochs {MASS_MLP_EPOCHS} || halt", 1),
            ('[ "${USED}" -lt 85 ]', f'[ "${{USED}}" -le {LINE} ]', 1),
            ("backoffLimitPerIndex: 6\n", "backoffLimitPerIndex: 2\n", 1),
            ("# GENERATED by scripts/build_atlas_res_job.py -- ATLAS jet-mass resolution (PI decision P7). CPU.",
             "# GENERATED by scripts/build_refit_v5_jobs.py (from build_atlas_res_job.spec) -- the v2 mass refit "
             "(PI 2026-11-14): alpha grid v2, MLP epochs recorded, one CPU model. CPU.", 1),
            ]
    for old, new, n in subs:
        if text.count(old) != n:
            raise SystemExit(f"FATAL: build_atlas_res_job.spec changed: {old!r} found {text.count(old)} times")
        text = text.replace(old, new)
    return pin_cpu(text)


def rebin_spec(tag: str) -> str:
    ba = _atlas_module()
    text = ba.rebin_spec(tag)
    subs = [(f"name: {ba.FINE_NAME}\n", "name: refit5-rebin-raunav\n", 1),
            (f"[ -f {ba.OUT_ROOT}/DONE ]", f"[ -f {ATLAS_V2}/DONE ]", 1),
            (f"{ba.OUT_ROOT} is not complete", f"{ATLAS_V2} is not complete", 1),
            (f"--in-dir {ba.OUT_ROOT} --out-dir {ba.FINE_ROOT}", f"--in-dir {ATLAS_V2} --out-dir {ATLAS_FINE_V2}", 1),
            (f"touch {ba.FINE_ROOT}/DONE", f"touch {ATLAS_FINE_V2}/DONE", 1)]
    for old, new, n in subs:
        if text.count(old) != n:
            raise SystemExit(f"FATAL: build_atlas_res_job.rebin_spec changed: {old!r} found {text.count(old)} times")
        text = text.replace(old, new)
    return text


def atlas_models() -> list[str]:
    return [m for m, *_ in _atlas_module().models()]


def _curves():
    return _load("build_curve_jobs", "scripts/build_curve_jobs.py")


def boot_spec(tag: str) -> str:
    """build_curve_jobs.boot_probe_v4_body's bootstrap on the v5 probes, top and quark/gluon from the fill probes
    at the chosen substitutions."""
    bc = _curves()
    models = bc.PROBE_MODELS
    pairs = bc.ratio_pairs([m.removeprefix("mtx-") for m, _ in models if m.startswith("mtx-")], "mtx-",
                           lambda k: f"init-s{k}")
    fn = bc.BOOT_PROBE_V4.format(models=" ".join(f"{m}:{ck}" for m, ck in models), probs=V5["fits_v5_probs"])
    # bootfill FILE: the same on <fill probes>/<model>/<checkpoint>.${FILL}/FILE.npz
    fn += ("          bootfill () {\n"
           "            local args=() m\n"
           "            for m in ${MODELS}; do args+=(--model \"${m%%:*}=" + V5["fits_v5_fill_probs"]
           + "/${m%%:*}/${m##*:}.${FILL}/$1.npz\"); done\n"
           "            python3 experiments/STATS/paired_bootstrap.py --task npz \"${args[@]}\" \"${PAIRS[@]}\" \\\n"
           "              --out ${TMP}/$1.json || halt\n"
           "          }\n")
    calls = [f"bootf jc1_N{bc.NMAX}", f"bootf jc2bal_N{bc.JC2BAL_SIZES[-1]}"] + [f"bootf jc2bal_pairs_{p}" for p in bc.PAIRS_188]
    calls += [f"FILL={FILL_TAG[ds]}; bootfill {ds}_N{n}" for ds in ("top", "qg", "qg_herwig") for n in bc.SIZES[ds]]
    inputs = [f"{V5['fits_v5']}/{bc._alt([m for m, _ in models])}.json",
              f"{V5['fits_v5_fill']}/{bc._alt([m for m, _ in models])}.json"]
    script = bc._guard([BOOT_V5], inputs) + bc._boot(BOOT_V5, pairs, fn, calls)
    body = "set -euo pipefail\n" + JB._start(tag) + "\n" + threads("4") + "\n" + guard() + "\n" + "\n".join(
        ln[10:] if ln.startswith("          ") else ln for ln in script.splitlines()) + '\necho "REFIT5 BOOT COMPLETE"\n'
    if "halt" in body:
        body = 'halt () { rc=$?; [ $rc -ge 128 ] && exit $rc; echo "HALT: exit $rc, not retried"; exit 42; }\n' + body
    return single("refit5-boot-raunav", body, "16Gi", "4",
                  "paired bootstrap over test jets of the v5 probes, top and quark/gluon under the chosen substitutions")


def bootl3_spec(tag: str) -> str:
    """decision L3's probe parts (build_boot_l3_jobs parts 2 and 7) on the v5 probes, into L3_V5."""
    bl = _load("build_boot_l3_jobs", "scripts/build_boot_l3_jobs.py")
    bl.OUT = L3_V5
    jc = [bl.pb("npz", bl.probes(V5["fits_v5_probs"], "best70", f), f"probe_{f}.json")
          for f in ("jc2bal_N1000000", "jc1_N1000000")]
    fill = [bl.pb("npz", bl.probes(V5["fits_v5_fill_probs"], f"best70.{FILL_TAG[ds]}", f"{ds}_N{n}"),
                  f"probe_{ds}_N{n}.{FILL_TAG[ds]}.json")
            for ds, n in (("top", 1_200_000), ("qg", 1_600_000), ("qg_herwig", 1_600_000))]
    runs = bl.runs()
    rows = ["\n    ".join(jc), "\n    ".join(fill)]
    body = f"""set -euo pipefail
{JB._start(tag)}
{threads(bl.CPU)}
{_wait([f"{V5['fits_v5']}/{m}.json" for m in runs] + [f"{V5['fits_v5_fill']}/{m}.json" for m in runs])}{guard()}
mkdir -p {L3_V5}
{case_block(rows)}echo "REFIT5 BOOT L3 PART ${{IDX}} COMPLETE"
"""
    return indexed("refit5-bootl3-raunav", body, len(rows), len(rows), bl.MEM, bl.CPU,
                   "decision L3's probe bootstraps on the v5 probes")


def check_families() -> list[tuple[str, list[str], bool]]:
    """(folder, expected models, one CPU model) of every refitted family."""
    ms = model_names()
    out = [(V5["fits_v5"], ms + [KIN], False), (V5["fits_v5_fill"], ms, False), (V5["fits_v5_sib2"], ms, False),
           (V5["fits_v5_coarse"], ms, False), (NAT, ms, False),
           (SOPHON_V5["fits_v5"], [JB.SOPHON_MODEL], False), (SOPHON_V5["fits_v5_fill"], [JB.SOPHON_MODEL], False)]
    xms = [m for m, _ in xbx_models()[0]]
    out += [(f"{XBX_V5}/{s}", xms, False) for s in XBX_SIGNALS]
    out += [(ATLAS_V2, atlas_models(), True), (MLP_V5, ms + [KIN], True)]
    return out


def checks_spec(tag: str) -> str:
    fams = check_families()
    waits = [f"{d}/{m}.json" for d, ms, _ in fams for m in ms] + [BOOT_V5, f"{L3_V5}/probe_top_N1200000.itop4.json",
                                                                   f"{ATLAS_FINE_V2}/DONE", f"{SOPHON_V5['fits_v5_qcdhf']}/{JB.SOPHON_MODEL}.json",
                                                                   f"{V5['fits_v5_qcdhf']}/init-s3.json"]
    name = lambda d: d.removeprefix("/data/results/eval/").replace("/", "_")  # noqa: E731
    calls = "".join(f"python3 experiments/EVAL/refit_checks.py --family {d} --expect {' '.join(ms)}"
                    f"{' --one-cpu' if one else ''} --out {CHECKS_ROOT}/{name(d)}.${{STAMP}}.json || FAIL=$((FAIL+1))\n"
                    for d, ms, one in fams)
    body = f"""set -euo pipefail
{JB._start(tag)}
{_wait(waits)}FAIL=0; STAMP=$(date -u +%Y%m%dT%H%M%SZ); mkdir -p {CHECKS_ROOT}
{calls}[ ${{FAIL}} = 0 ] || {{ echo "FATAL: ${{FAIL}} families fail the refit rules ({CHECKS_ROOT}/*.${{STAMP}}.json)"; exit 42; }}
touch {CHECKS_ROOT}/PASS        # experiments/FIGS/pull_mlst_data.py pulls the v5 families only after this
echo "REFIT5 CHECKS PASS"
"""
    return single("refit5-checks-raunav", body, "4Gi", "1", "the refit rules on every v5 family (PI 2026-11-14)")


def build(tag: str, parallelism: int) -> dict[str, str]:
    return {"fit": fit_spec(tag, parallelism), "fill": fill_spec(tag, parallelism), "sib": sib_spec(tag, parallelism),
            "coarse": coarse_spec(tag, parallelism), "qcdhf": qcdhf_spec(tag), "nat": nat_spec(tag, parallelism),
            "sophonx": sophonx_spec(tag), "sophon": sophon_spec(tag), "xbx": xbx_spec(tag, parallelism),
            "mlp": mlp_spec(tag, parallelism), "atlas": atlas_spec(tag, parallelism), "rebin": rebin_spec(tag),
            "boot": boot_spec(tag), "bootl3": bootl3_spec(tag), "checks": checks_spec(tag)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", required=True, help="the pushed tag the pods clone")
    ap.add_argument("--parallelism", type=int, default=5)
    ap.add_argument("kinds", nargs="+", choices=("all", "fit", "fill", "sib", "coarse", "qcdhf", "nat", "sophonx",
                                                 "sophon", "xbx", "mlp", "atlas", "rebin", "boot", "bootl3", "checks"))
    a = ap.parse_args(argv)
    if not re.fullmatch(r"mtx-s\d+\.\d+", a.tag):
        raise SystemExit(f"FATAL: --tag {a.tag} is not an mtx-sN.NN tag")
    specs = build(a.tag, a.parallelism)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for kind in (specs if "all" in a.kinds else a.kinds):
        p = OUT_DIR / f"job-refit5-{kind}-raunav.yaml"
        p.write_text(specs[kind])
        print(p.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
