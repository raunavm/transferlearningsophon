"""Jobs for the downstream linear probes of the v2 models (PI, 2026-10-09: "linear probe and full
fine tuning for the Top tagging and the 2 Q/G and jetclass").

Per model (every run of configs/arms/v2_grid.json, and the untrained trunks init-s1..3):

  linprobe-x-<model>   GPU, any product but the RTX 3090 (the grid's), strict fp32.
      experiments/EVAL/extract_v2.py writes the frozen class-token features of the primary
      checkpoint and its BatchNorm twin (best70 best70_bn; the untrained trunk: init) for every
      split fine-tuning reads: the seed-1 training subsets at fine-tuning's sizes, validation,
      test, and the Herwig quark/gluon test. One pass per split serves both checkpoints.
  linprobe-fit-<model> GPU, the same products. experiments/EVAL/linear_probe_v2.py fits and scores the probes;
      experiments/EVAL/mass_resolution.py regresses the jet mass from the JetClass-II test
      features (the first 2M jets are test2m_observers' jets, label digest checked).

A run's x job needs its BatchNorm twin, so it is applied after the run's twin job; the fit
job after its x job. Features go to LP_ROOT/<model>/<dataset>/<split>/<checkpoint>/ (float16,
about 2.9 GB a checkpoint of a run), fits to LP_ROOT/FITS/<model>.json.

--v3 (2026-10-29, the JetClass-II task at 188 classes) writes two Indexed Jobs, whose pods the
Job controller releases PARALLEL at a time (no release loop), one pod per model:
  linprobe-fit3-raunav  the refit of the 27 paper runs and the 3 untrained trunks: jc2_188 jc1
      top qg into LP_ROOT/FITS_V3, each cell's probe and test outputs into LP_ROOT/PROBS_V3
  linprobe-mlp3-raunav  CPU, the hierarchy's 12 runs and the untrained trunks: the nonlinear
      probe on the binary tasks at their largest training set (D6) into LP_ROOT/MLP_V3

--v4 (2026-10-30: the nonlinear probe beside every null; the untrained trunk with its BatchNorm
statistics recomputed, init_bn) writes three more Indexed Jobs:
  linprobe-mlp4-raunav  CPU, as mlp3: the nonlinear probe's v4 cells (the binary tasks, jc1 and
      jc2_188 at their largest training set) of the 27 paper runs at best70 and the untrained
      trunks at init, into LP_ROOT/MLP_V4/<model>.json. Applied now: its features exist.
  linprobe-initbn-raunav  GPU, any product: fit3's linear probes of init-s1..3 at init_bn into
      LP_ROOT/FITS_INITBN, probes and test outputs into LP_ROOT/PROBS_INITBN
  linprobe-mlp4bn-raunav  CPU, as mlp4: the nonlinear probe of init-s1..3 at init_bn into
      LP_ROOT/MLP_V4/init_bn/<model>.json (its own job, so no GPU waits on CPU fits)
Apply initbn and mlp4bn only after the init_bn extraction jobs init-bn-s1..s3 have completed:
neither waits for its features (a pod waiting would hold its node), and linear_probe_v2.py
stops on a missing split.

    python3 scripts/build_linprobe_jobs.py [--pin-not-yet-tagged] [--only mtx-l188-s1 ...] [--v3 | --v4]
"""
from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "experiments" / "EVAL" / "k8s" / "v2" / "linprobe"
LP_PIN = "mtx-s2.03"         # extraction (extract_v2.py --no-pooled)
# fitting (linear_probe_v2.py, torch L-BFGS to 3000 iterations, TF32 recorded; mass_resolution.py):
# mtx-s2.08 for every fit run before 2026-10-27; mtx-s2.12 adds only eval_arm's tie for a jet whose
# two probabilities round to 0 (which would have stopped a fit), so its fits equal mtx-s2.08's
FIT_PIN = "mtx-s2.12"
LP_ROOT = "/data/results/eval/v2_linprobe"
FITS = "fits3000"   # fits at MAX_ITER 3000; LP_ROOT/fits holds the mtx-s2.06 smoke fit (1000, short of convergence)
MASS_OBS = "/data/results/eval/test2m_observers"   # generator-level groomed mass of TEST2M
V3_PIN = "mtx-s2.18"
FITS_V3, PROBS_V3, MLP_V3 = "fits_v3", "fits_v3_probs", "fits_v3_mlp"
V3_DATASETS = ("jc2_188", "jc1", "top", "qg")
# the refit may take any GPU, the RTX 3090 included: the grid it was kept free for has finished
# (2026-10-29), and fine-tuning runs on V100s
V3_NEEDED = {"experiments/EVAL/linear_probe_v2.py": "def native_metrics"}
OUT_DIR_V3 = ROOT / "experiments" / "EVAL" / "k8s" / "v2" / "linprobe_v3"
RUN_CHECKPOINTS = ("best70", "best70_bn")
INIT_CHECKPOINTS = ("init",)
NOT_ON = ("NVIDIA-GeForce-RTX-3090",)      # the v2 grid's and fine-tuning's product
# weaver loads a whole file per fetch, and a 1M-jet training subset is one file: fine-tuning
# reads the same files at 88Gi; at 32Gi the first smoke test (2026-10-09) was OOM-killed
# twice on JetClass-II train_N1000000
X_MEM = "88Gi"
# What each tag must carry for what its jobs run.
NEEDED = {"experiments/EVAL/extract_v2.py": "--no-pooled"}
FIT_NEEDED = {"experiments/EVAL/linear_probe_v2.py": "\"tf32\": tf32",
              "experiments/EVAL/mass_resolution.py": "n_max: int | None = None",
              "experiments/EVAL/eval_arm.py": "scores the tie"}


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


BX = _load("build_extract_jobs", "scripts/build_extract_jobs.py")
BF = _load("build_ft_jobs", "scripts/build_ft_jobs.py")

# dataset -> (training sizes, data config, {split: (--max-jets, files)}). The training sizes,
# subsets and test sets are fine-tuning's (scripts/build_ft_jobs.py); "$TEST1" is leg 2's
# 20 JetClass test files, listed in the pod as leg 2 lists them.
JC2, JC1, TOP, QG = "/data/finetune/jc2_v2", "/data/finetune/jc1", "/data/finetune/top_sub", "/data/finetune/qg_v2_sub"
DATASETS = {
    "jc2": (BF.SIZES, "configs/finetune/JetClassII_L162_noweight.yaml",
            {"val": (0, f"{JC2}/val.parquet"),
             # leg 1's test: TEST2M with native labels, the first 2M jets
             "test": (2_000_000, "configs/data/JetClassII_base.yaml", BF.test2m_list())}),
    "jc1": (BF.SIZES, "configs/finetune/JetClassI_sophon_noweight.yaml",
            {"val": (0, f"{JC1}/val.parquet"), "test": (0, "${TEST1}")}),
    "top": (BF.BENCH_SIZES["top"], "configs/finetune/TopReference.yaml",
            {"val": (0, f"{TOP}/val.parquet"), "test": (0, "/data/finetune/top/top_test.parquet")}),
    "qg": (BF.BENCH_SIZES["qg"], "configs/finetune/EnergyFlowQG.yaml",
           {"val": (0, f"{QG}/val.parquet"),
            "test": (0, "/data/finetune/qg_v2/qg_chunk18.parquet /data/finetune/qg_v2/qg_chunk19.parquet"),
            "herwig": (0, "/data/finetune/qg_herwig/qg_herwig_chunk0.parquet "
                          "/data/finetune/qg_herwig/qg_herwig_chunk1.parquet")}),
}
TRAIN_DIR = {"jc2": JC2, "jc1": JC1, "top": TOP, "qg": QG}

LIST_TEST1 = """          TEST1=""
          for C in {classes}; do
            for f in $(ls /data/JetClass/Pythia/test_20M/${{C}}_*.root | sort | head -2); do TEST1="${{TEST1}} ${{f}}"; done
          done
          [ "$(echo ${{TEST1}} | wc -w)" -eq 20 ] || {{ echo "FATAL: expected 20 JetClass test files"; exit 42; }}
"""

EXTRACT_FN = """          # extract DATASET SPLIT CONFIG MAX_JETS FILES...: one pass, every checkpoint
          extract () {{
            OUT={root}/{model}/$1/$2; CFG=$3; MAXJ=$4; shift 4
          python3 experiments/EVAL/extract_v2.py \\
              --run-dir {run_dir} --rung {rung} --num-classes {k} --num-reg {reg} \\
              --checkpoints {ckpts} --data-config ${{CFG}} --max-jets ${{MAXJ}} \\
              --observers --feature-classes --prefix-features 2000000000 \\
              --head-prefix 0 --no-anomaly-rows --diag-stride 0 --no-pooled \\
              --data-test "$@" \\
              --out ${{OUT}} || halt
          }}
"""


# Jets a split holds besides the training subsets (fine-tuning's, scripts/build_ft_jobs.py):
# validation sets 20,480 (JetClass-II) and 200,000 (JetClass 2e4 per class; top; q/g chunks
# 16-17); test sets 2M (+ the end of the last batch, JetClass-II), 2M, 404,000, 200,000 and
# Herwig 200,000. A row is 128 float16 features, an int16 label and an int64 row index.
OTHER_JETS = {"jc2": 20_480 + 2_000_512, "jc1": 200_000 + 2_000_000, "top": 200_000 + 404_000,
              "qg": 200_000 + 200_000 + 200_000}
ROW_BYTES = 128 * 2 + 2 + 8


def planned_bytes() -> int:
    """What every x job writes to /data: each model's rows at each of its checkpoints."""
    per_ckpt = sum(sum(DATASETS[ds][0]) + OTHER_JETS[ds] for ds in DATASETS) * ROW_BYTES
    return sum(len(ckpts) for *_, ckpts in models()) * per_ckpt


def models() -> list[tuple[str, str, str, int, int, tuple]]:
    """(model, run directory, rung, classes, regression outputs, checkpoints)."""
    out = [(run, run, BX.v2_rung(arm), k, reg, RUN_CHECKPOINTS) for run, arm, k, reg, _s in BX.v2_runs()]
    out += [(ref, run, "none", 0, 0, INIT_CHECKPOINTS) for ref, run in BX.v2_init_refs()]
    return out


def calls() -> list[str]:
    """The extract calls of one model, every split of every dataset."""
    lines = []
    for ds, (sizes, cfg, extra) in DATASETS.items():
        for n in sizes:
            lines.append(f"          extract {ds} train_N{n} {cfg} 0 {TRAIN_DIR[ds]}/train_N{n}_s1.parquet\n")
        for split, spec in extra.items():
            maxj, *rest = spec
            c, files = (rest if len(rest) == 2 else (cfg, rest[0]))
            lines.append(f"          extract {ds} {split} {c} {maxj} {files}\n")
    return lines


def off_grid_gpu(text: str, extract: bool = True, not_on: tuple = NOT_ON) -> str:
    """A GPU job spec moved off the grid's product, with build_extract_jobs' GPU-fault rule
    (for a job that runs no extraction, its GPU check, halt and node exclusions only)."""
    if extract:
        text = BX.gpu_fault_aware(text)
    else:
        for old, new in ((BX.HALT, BX.HALT_GPU), (BX.GPU_CHECK.format(), BX.GPU_OK),
                         (BX.NODE_EXCLUDE, BX.NODE_EXCLUDE.split("values: [")[0] + "values: ["
                          + ", ".join(f'"{n}"' for n in BX.GPU_FAULT_NODES) + "]")):
            assert text.count(old) == 1, old[:60]
            text = text.replace(old, new)
    old = ", ".join(f'"{n}"' for n in BX.GPU_FAULT_NODES)
    assert text.count(old) == 1
    text = text.replace(old, ", ".join(f'"{n}"' for n in BX.V2_GPU_FAULT_NODES))
    if not not_on:
        return text
    product = ("\n              - key: nvidia.com/gpu.product\n                operator: NotIn\n"
               "                values: [" + ", ".join(f'"{p}"' for p in not_on) + "]")
    assert text.count("\n      volumes:\n") == 1
    return text.replace("\n      volumes:\n", product + "\n      volumes:\n")


def x_spec(model, run, rung, k, reg, ckpts) -> str:
    body = (LIST_TEST1.format(classes=BF.JC1_CLASSES)
            + EXTRACT_FN.format(root=LP_ROOT, model=model, run_dir=f"{BX.V2_ROOT}/{run}", rung=rung,
                                k=k, reg=reg, ckpts=" ".join(ckpts))
            + "".join(calls())
            + '          echo "every split of ' + model + ' extracted"\n')
    return off_grid_gpu(BX.V1ERR_TEMPLATE.format(
        name=f"linprobe-x-{model.removeprefix('mtx-')}-raunav", image=BX.IMAGE, pin=LP_PIN, body=body,
        mem=X_MEM, cpu="4", gpu_req=', nvidia.com/gpu: "1"', gpu_check=BX.GPU_CHECK.format(),
        node_exclude=BX.NODE_EXCLUDE))


def fit_spec(model, ckpts) -> str:
    body = (f"          python3 experiments/EVAL/linear_probe_v2.py --root {LP_ROOT} --model {model} \\\n"
            f"            --checkpoints {' '.join(ckpts)} --out {LP_ROOT}/{FITS} || halt\n"
            f"          python3 experiments/EVAL/mass_resolution.py --observers {MASS_OBS} \\\n"
            f"            --features {' '.join(f'{c}={LP_ROOT}/{model}/jc2/test/{c}' for c in ckpts)} \\\n"
            f"            --out {LP_ROOT}/mass/{model} || halt\n")
    # the probes are full-batch L-BFGS in torch on the GPU (linear_probe_v2.Logit)
    return off_grid_gpu(BX.V1ERR_TEMPLATE.format(
        name=f"linprobe-fit-{model.removeprefix('mtx-')}-raunav", image=BX.IMAGE, pin=FIT_PIN, body=body,
        mem="32Gi", cpu="8", gpu_req=', nvidia.com/gpu: "1"', gpu_check=BX.GPU_CHECK.format(),
        node_exclude=BX.NODE_EXCLUDE), extract=False)


V3_PARALLEL = {"fit3": 6, "mlp3": 5}


def indexed(text: str, n: int, parallel: int) -> str:
    """A V1ERR_TEMPLATE job made Indexed: n completions, at most `parallel` pods at a time,
    each index retried on its own (a signal) and failed alone on exit 42."""
    old_b, old_p = "  backoffLimit: 6\n", "    - action: FailJob\n"
    assert text.count(old_b) == 1 and text.count(old_p) == 1
    return (text.replace(old_b, f"  completionMode: Indexed\n  completions: {n}\n  parallelism: {parallel}\n"
                                f"  backoffLimitPerIndex: 6\n  maxFailedIndexes: {n}\n")
                .replace(old_p, "    - action: FailIndex\n"))


def pick(models: list[tuple[str, tuple]]) -> str:
    """Bash that sets MODEL and CKPTS from the pod's completion index."""
    return ("          MODELS=(" + " ".join(m for m, _ in models) + ")\n"
            "          CKPTS=(" + " ".join('"' + " ".join(c) + '"' for _, c in models) + ")\n"
            "          MODEL=${MODELS[$JOB_COMPLETION_INDEX]}; CKPT=${CKPTS[$JOB_COMPLETION_INDEX]}\n"
            '          echo "index $JOB_COMPLETION_INDEX: $MODEL ($CKPT)"\n')


def v3_models() -> list[tuple[str, tuple]]:
    """The 27 paper runs and the untrained trunks, in the paper's order."""
    paper = BF.v2_paper_runs()
    have = {m: c for m, *_, c in models()}
    return [(m, have[m]) for m in paper] + [(m, c) for m, c in have.items() if m.startswith("init-")]


def fit_v3_spec(ms) -> str:
    body = (pick(ms)
            + f"          python3 experiments/EVAL/linear_probe_v2.py --root {LP_ROOT} --model $MODEL \\\n"
            f"            --checkpoints $CKPT --datasets {' '.join(V3_DATASETS)} \\\n"
            f"            --out {LP_ROOT}/{FITS_V3} --probs-out {LP_ROOT}/{PROBS_V3} || halt\n")
    return indexed(off_grid_gpu(BX.V1ERR_TEMPLATE.format(
        name="linprobe-fit3-raunav", image=BX.IMAGE, pin=V3_PIN, body=body,
        mem="48Gi", cpu="8", gpu_req=', nvidia.com/gpu: "1"', gpu_check=BX.GPU_CHECK.format(),
        node_exclude=BX.NODE_EXCLUDE), extract=False, not_on=()), len(ms), V3_PARALLEL["fit3"])


def mlp_spec(ms) -> str:
    """CPU only: probe.fit_mlp pins torch to its own 4 threads."""
    body = (pick([(m, c[:1]) for m, c in ms])
            + f"          python3 experiments/EVAL/linear_probe_v2.py --root {LP_ROOT} --model $MODEL \\\n"
            f"            --checkpoints $CKPT --mlp --out {LP_ROOT}/{MLP_V3} || halt\n")
    return indexed(BX.V1ERR_TEMPLATE.format(
        name="linprobe-mlp3-raunav", image=BX.IMAGE, pin=V3_PIN, body=body,
        mem="48Gi", cpu="4", gpu_req="", gpu_check="", node_exclude=BX.NODE_EXCLUDE),
        len(ms), V3_PARALLEL["mlp3"])


def hierarchy(model: str) -> bool:
    """The four levels of the class hierarchy and the untrained trunks (not mass, partitions)."""
    return model.startswith("init-") or (model.split("-")[1] in ("l188", "l162", "r42q1", "r16q1"))


def build_v3(only=None) -> dict[str, str]:
    ms = [x for x in v3_models() if not only or x[0] in only]
    return {"job-linprobe-fit3-raunav.yaml": fit_v3_spec(ms),
            "job-linprobe-mlp3-raunav.yaml": mlp_spec([x for x in ms if hierarchy(x[0])])}


V4_PIN = "mtx-s2.20"
MLP_V4, FITS_INITBN, PROBS_INITBN = "fits_v3_mlp2", "fits_v3_initbn", "fits_v3_initbn_probs"
V4_NEEDED = {"experiments/EVAL/linear_probe_v2.py": '"v4": ("jc2_188", "jc1", "top", "qg")',
             "experiments/EVAL/probe.py": "Xte, seeds=MLP_SEEDS, n_classes=2"}
OUT_DIR_V4 = ROOT / "experiments" / "EVAL" / "k8s" / "v2" / "linprobe_v4"
V4_PARALLEL = {"mlp4": 2, "initbn": 1, "mlp4bn": 2}
# the V100s, which the 188-class fine-tuning waits for (2026-10-30: every one in use), stay free
INITBN_NOT_ON = ("Tesla-V100-SXM2-32GB",)


def v4_mlp_models() -> list[tuple[str, tuple]]:
    """The 27 paper runs at best70 and the untrained trunks at init."""
    return [(m, c[:1]) for m, c in v3_models()]


def initbn_models() -> list[tuple[str, tuple]]:
    return [(m, ("init_bn",)) for m, _ in BX.v2_init_refs()]


def mlp_v4_spec(ms, name: str = "mlp4", out: str = MLP_V4) -> str:
    """CPU only, as mlp_spec: probe.fit_mlp pins torch to its own 4 threads."""
    body = (pick(ms)
            + f"          python3 experiments/EVAL/linear_probe_v2.py --root {LP_ROOT} --model $MODEL \\\n"
            f"            --checkpoints $CKPT --mlp v4 --out {LP_ROOT}/{out} || halt\n")
    return indexed(BX.V1ERR_TEMPLATE.format(
        name=f"linprobe-{name}-raunav", image=BX.IMAGE, pin=V4_PIN, body=body,
        mem="48Gi", cpu="4", gpu_req="", gpu_check="", node_exclude=BX.NODE_EXCLUDE),
        len(ms), V4_PARALLEL[name])


def initbn_spec(ms) -> str:
    """fit_v3_spec's fits of the init_bn checkpoint, into their own directories."""
    body = (pick(ms)
            + f"          python3 experiments/EVAL/linear_probe_v2.py --root {LP_ROOT} --model $MODEL \\\n"
            f"            --checkpoints $CKPT --datasets {' '.join(V3_DATASETS)} \\\n"
            f"            --out {LP_ROOT}/{FITS_INITBN} --probs-out {LP_ROOT}/{PROBS_INITBN} || halt\n")
    return indexed(off_grid_gpu(BX.V1ERR_TEMPLATE.format(
        name="linprobe-initbn-raunav", image=BX.IMAGE, pin=V4_PIN, body=body,
        mem="48Gi", cpu="8", gpu_req=', nvidia.com/gpu: "1"', gpu_check=BX.GPU_CHECK.format(),
        node_exclude=BX.NODE_EXCLUDE), extract=False, not_on=INITBN_NOT_ON), len(ms), V4_PARALLEL["initbn"])


def build_v4(only=None) -> dict[str, str]:
    keep = lambda ms: [x for x in ms if not only or x[0] in only]
    return {"job-linprobe-mlp4-raunav.yaml": mlp_v4_spec(keep(v4_mlp_models())),
            "job-linprobe-initbn-raunav.yaml": initbn_spec(keep(initbn_models())),
            "job-linprobe-mlp4bn-raunav.yaml": mlp_v4_spec(keep(initbn_models()), "mlp4bn",
                                                           f"{MLP_V4}/init_bn")}


def build(only=None) -> dict[str, str]:
    specs = {}
    for model, run, rung, k, reg, ckpts in models():
        if only and model not in only:
            continue
        stem = model.removeprefix("mtx-")
        specs[f"job-linprobe-x-{stem}-raunav.yaml"] = x_spec(model, run, rung, k, reg, ckpts)
        specs[f"job-linprobe-fit-{stem}-raunav.yaml"] = fit_spec(model, ckpts)
    return specs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pin-not-yet-tagged", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--v3", action="store_true", help="the 2026-10-29 refit and MLP probes")
    ap.add_argument("--v4", action="store_true", help="the 2026-10-30 MLP probes and init_bn fits")
    a = ap.parse_args(argv)
    if a.v3:
        BX.verify_pin(V3_PIN, list(V3_NEEDED), a.pin_not_yet_tagged, V3_NEEDED)
        specs, out_dir = build_v3(a.only), OUT_DIR_V3
    elif a.v4:
        BX.verify_pin(V4_PIN, list(V4_NEEDED), a.pin_not_yet_tagged, V4_NEEDED)
        specs, out_dir = build_v4(a.only), OUT_DIR_V4
    else:
        BX.verify_pin(LP_PIN, list(NEEDED), False, NEEDED)
        BX.verify_pin(FIT_PIN, list(FIT_NEEDED), a.pin_not_yet_tagged, FIT_NEEDED)
        specs, out_dir = build(a.only), OUT_DIR
    if not a.check_only:
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, text in specs.items():
            (out_dir / name).write_text(text)
    print(f"{len(specs)} specs {'checked' if a.check_only else 'written to ' + str(out_dir.relative_to(ROOT))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
