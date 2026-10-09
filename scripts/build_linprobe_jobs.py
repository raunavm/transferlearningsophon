"""Jobs for the downstream linear probes of the v2 models (PI, 2026-10-09: "linear probe and full
fine tuning for the Top tagging and the 2 Q/G and jetclass").

Per model (every run of configs/arms/v2_grid.json, and the untrained trunks init-s1..3):

  linprobe-x-<model>   GPU, any product but the RTX 3090 (the grid's), strict fp32.
      experiments/EVAL/extract_v2.py writes the frozen class-token features of the primary
      checkpoint and its BatchNorm twin (best70 best70_bn; the untrained trunk: init) for every
      split fine-tuning reads: the seed-1 training subsets at fine-tuning's sizes, validation,
      test, and the Herwig quark/gluon test. One pass per split serves both checkpoints.
  linprobe-fit-<model> CPU. experiments/EVAL/linear_probe_v2.py fits and scores the probes.

A run's x job needs its BatchNorm twin, so it is applied after the run's twin job; the fit
job after its x job. Features go to LP_ROOT/<model>/<dataset>/<split>/<checkpoint>/ (float16,
about 2.9 GB a checkpoint of a run), fits to LP_ROOT/fits/<model>.json.

    python3 scripts/build_linprobe_jobs.py [--pin-not-yet-tagged] [--only mtx-l188-s1 ...]
"""
from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "experiments" / "EVAL" / "k8s" / "v2" / "linprobe"
LP_PIN = "mtx-s2.03"
LP_ROOT = "/data/results/eval/v2_linprobe"
RUN_CHECKPOINTS = ("best70", "best70_bn")
INIT_CHECKPOINTS = ("init",)
NOT_ON = ("NVIDIA-GeForce-RTX-3090",)      # the v2 grid's and fine-tuning's product
# What the tag must carry for the flags these jobs pass.
NEEDED = {"experiments/EVAL/extract_v2.py": "--no-pooled",
          "experiments/EVAL/linear_probe_v2.py": "def probe_cells"}


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


def x_spec(model, run, rung, k, reg, ckpts) -> str:
    body = (LIST_TEST1.format(classes=BF.JC1_CLASSES)
            + EXTRACT_FN.format(root=LP_ROOT, model=model, run_dir=f"{BX.V2_ROOT}/{run}", rung=rung,
                                k=k, reg=reg, ckpts=" ".join(ckpts))
            + "".join(calls())
            + '          echo "every split of ' + model + ' extracted"\n')
    text = BX.V1ERR_TEMPLATE.format(name=f"linprobe-x-{model.removeprefix('mtx-')}-raunav", image=BX.IMAGE,
                                    pin=LP_PIN, body=body, mem="32Gi", cpu="4",
                                    gpu_req=', nvidia.com/gpu: "1"', gpu_check=BX.GPU_CHECK.format(),
                                    node_exclude=BX.NODE_EXCLUDE)
    text = BX.gpu_fault_aware(text)
    old = ", ".join(f'"{n}"' for n in BX.GPU_FAULT_NODES)
    assert text.count(old) == 1
    text = text.replace(old, ", ".join(f'"{n}"' for n in BX.V2_GPU_FAULT_NODES))
    product = ("\n              - key: nvidia.com/gpu.product\n                operator: NotIn\n"
               "                values: [" + ", ".join(f'"{p}"' for p in NOT_ON) + "]")
    assert text.count("\n      volumes:\n") == 1
    return text.replace("\n      volumes:\n", product + "\n      volumes:\n")


def fit_spec(model, ckpts) -> str:
    body = (f"          python3 experiments/EVAL/linear_probe_v2.py --root {LP_ROOT} --model {model} \\\n"
            f"            --checkpoints {' '.join(ckpts)} --out {LP_ROOT}/fits || halt\n")
    return BX.V1ERR_TEMPLATE.format(name=f"linprobe-fit-{model.removeprefix('mtx-')}-raunav", image=BX.IMAGE,
                                    pin=LP_PIN, body=body, mem="32Gi", cpu="8", gpu_req="", gpu_check="",
                                    node_exclude=BX.NODE_EXCLUDE)


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
    a = ap.parse_args(argv)
    BX.verify_pin(LP_PIN, list(NEEDED), a.pin_not_yet_tagged, NEEDED)
    specs = build(a.only)
    if not a.check_only:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        for name, text in specs.items():
            (OUT_DIR / name).write_text(text)
    print(f"{len(specs)} specs {'checked' if a.check_only else 'written to ' + str(OUT_DIR.relative_to(ROOT))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
