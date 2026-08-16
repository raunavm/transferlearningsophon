#!/usr/bin/env python3
"""Emit experiments/EVAL/k8s/job-eval-<run>-raunav.yaml -- frozen-checkpoint eval.

WHY THIS IS A SEPARATE JOB
--------------------------
The training pods deliberately pass no `--data-test` / `--predict-output`.
weaver accumulates EVERY test score in RAM and then concatenates: over the
27,448,839 selected test jets that is 17.8 GB of scores and a ~35.6 GB peak at
K=162, on top of a training process whose measured anon band is 35-58 GB. Doing
it in-pod would OOM L162 at roughly day 8 -- after the entire budget was spent.
It is also an I1 hazard, because the failure probability would scale with K,
i.e. along the studied variable.

`docs/GATES.md` still requires the cached ROC arrays ("a run without cached ROC
arrays is a failed run even if its AUC is right"). This job supplies them, from
the frozen `net_best_epoch_state.pt`, exactly as E1 did.

THREE THINGS THAT ARE EASY TO GET WRONG, ALL VERIFIED AGAINST weaver 0.4.17
--------------------------------------------------------------------------
1. `--predict-gpus 0` IS MANDATORY. In `train.py`, the eval branch reads:
       if args.predict_gpus: dev = torch.device(gpus[0])
       else:                 dev = torch.device('cpu')
   Omit it and weaver silently evaluates 27.4 M jets ON CPU. It does not warn,
   it does not fail -- it just takes days.

2. `--data-test` MUST be UNPREFIXED. On `--data-train` the `Res2P:`/`Res34P:`/
   `QCD:` prefix is how weaver groups files for reweighting; on `--data-test` it
   names an evaluation GROUP, and weaver then renames the output per group --
   `pred_Res2P.root`, `pred_Res34P.root`, `pred_QCD.root` -- and never writes
   `pred.root` at all. Unprefixed gives one combined evaluation over all 335
   test files, which is what the ROC analysis wants.

3. Checkpoint path is `<model-prefix>_best_epoch_state.pt` unless the prefix
   itself ends in `.pt`. So `--model-prefix <run>/net` loads
   `<run>/net_best_epoch_state.pt`, which is weaver's own best-epoch selection.

NOT REWEIGHTED, AND THAT IS CORRECT
-----------------------------------
`test_load` builds `SimpleIterDataset(..., for_training=False, ...)`, so the flat
(pT, m_SD) resampling used in training is NOT applied here. AUC is measured on
the natural test distribution, which is what the paper reports.

NO GPU-MODEL PIN, DELIBERATELY
------------------------------
I7 constrains TRAINING: two arms of a seed pair must share a GPU model or the
pairing gain is silently lost. Evaluation is deterministic inference from a
frozen checkpoint, so cross-GPU differences are far below the reported
precision. Leaving the model unpinned widens the schedulable pool, which
matters because this job wants ~120Gi.

Run:  python3 scripts/build_eval_jobs.py <gate> <run_id> <ARM> <K> [--pin REF]
  e.g. python3 scripts/build_eval_jobs.py mtx mtx-l162-s1 L162 162
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "experiments/MTX/k8s/job-mtx-l162-s1-raunav.yaml"
OUT_DIR = ROOT / "experiments/EVAL/k8s"

# 17.8 GB of scores + ~35.6 GB concatenate peak at K=162, plus the loader band.
EVAL_MEMORY = "120Gi"


def build(gate: str, run_id: str, arm: str, k: int, pin: str) -> tuple[str, str]:
    text = SRC.read_text()
    name = f"eval-{run_id}"

    # identity
    text = re.sub(r"^  name: mtx-\S+\n", f"  name: {name}-raunav\n", text, flags=re.M)
    text = re.sub(r"^          RUN_ID=\S+\n", f"          RUN_ID={run_id}\n",
                  text, flags=re.M)
    text = text.replace("OUT=/data/results/mtx/${RUN_ID}",
                        f"OUT=/data/results/{gate}/${{RUN_ID}}/eval")
    text = text.replace("configs/arms/L162.yaml", f"configs/arms/{arm}.yaml")
    text = text.replace("configs/arms/L162.", f"configs/arms/{arm}.")
    text = text.replace("/data/results/mtx/makeweight/L162.",
                        f"/data/results/mtx/makeweight/{arm}.")
    text = re.sub(r'value: "mtx-s1[^"]*"', f'value: "{pin}"', text)

    # resources: more memory, no GPU-model pin (see module docstring)
    text = re.sub(r'memory: "\d+Gi"', f'memory: "{EVAL_MEMORY}"', text)
    text = re.sub(r"\n              - key: nvidia\.com/gpu\.product\n"
                  r"                operator: In\n"
                  r'                values: \["[^"]*"\]', "", text)

    # replace the whole training invocation with the predict invocation
    start = text.index("          # Manifest FIRST")
    end = text.index("          cp -r ./runs")
    ckpt = f"/data/results/{gate}/{run_id}/net"
    test_files = ("/jc2/jet_data/Res2P_{0250..0299}.parquet "
                  "/jc2/jet_data/Res34P_{1075..1289}.parquet "
                  "/jc2/jet_data/QCD_{0350..0419}.parquet")
    block = f"""          CKPT={ckpt}_best_epoch_state.pt
          [ -s "${{CKPT}}" ] || {{ echo "FATAL: no checkpoint at ${{CKPT}}"; exit 1; }}
          echo "evaluating ${{CKPT}}"

          # --predict sets run_mode=['test'] and skips training entirely.
          # --predict-gpus 0 is MANDATORY: without it weaver evaluates on CPU,
          # silently, over 27.4 M jets. See this script's docstring.
          # --data-test is UNPREFIXED so weaver writes pred.root rather than
          # pred_<group>.root.
          PYTHONUNBUFFERED=1 weaver --predict --predict-gpus 0 \\
            --data-test {test_files} \\
            --data-config configs/arms/{arm}.yaml \\
            --no-remake-weights \\
            --network-config experiments/MTX/ParT_sophon_arch_mtx.py \\
            -o num_classes {k} -o fc_params '[(512,0.1)]' \\
            --batch-size 512 \\
            --num-workers 2 --fetch-by-files --fetch-step 5 \\
            --model-prefix {ckpt} \\
            --log ${{OUT}}/eval.log \\
            --predict-output ${{OUT}}/pred.root

          # pred.root is the gate requirement, so verify it exists AND is a real
          # ROOT file. weaver wraps its writer in try/except and only LOGS a
          # failure; uproot.recreate has already created the file by then, so a
          # size check alone would pass on a corrupt write.
          # Kept to ONE line on purpose: the YAML block scalar above requires
          # every line to stay indented, and a multi-line `python3 -c` with
          # column-0 lines silently terminates the block and breaks the spec.
          [ -s ${{OUT}}/pred.root ] || {{ echo "FATAL: no pred.root written"; exit 1; }}
          python3 -c "import uproot,sys; f=uproot.open('${{OUT}}/pred.root'); t=f[f.keys()[0].split(';')[0]]; print('pred.root OK:', t.name, t.num_entries, 'entries,', len(t.keys()), 'branches'); sys.exit(0 if t.num_entries>0 else 1)" || {{ echo "FATAL: pred.root unreadable or empty"; exit 1; }}
"""
    text = text[:start] + block + text[end:]
    return f"job-{name}-raunav.yaml", text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gate")
    ap.add_argument("run_id")
    ap.add_argument("arm")
    ap.add_argument("k", type=int)
    ap.add_argument("--pin", default="mtx-s1.1")
    a = ap.parse_args()

    if not SRC.exists():
        sys.exit(f"FATAL: {SRC} not found")
    fname, text = build(a.gate, a.run_id, a.arm, a.k, a.pin)

    d = yaml.safe_load(text)
    name = d["metadata"]["name"]
    if not re.fullmatch(r"[a-z0-9]([-a-z0-9.]*[a-z0-9])?", name):
        sys.exit(f"FATAL: {name} is not a valid RFC 1123 name")
    args_block = d["spec"]["template"]["spec"]["containers"][0]["args"][0]
    for must in ("--predict ", "--predict-gpus 0", "--predict-output",
                 f"-o num_classes {a.k}"):
        if must not in args_block:
            sys.exit(f"FATAL: generated spec is missing {must!r}")
    if "Res2P:" in args_block.split("--data-test")[1].split("\\")[0]:
        sys.exit("FATAL: --data-test is prefixed; weaver would not write pred.root")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / fname).write_text(text)
    print(f"wrote {(OUT_DIR / fname).relative_to(ROOT)}")
    print(f"  job={name} arm={a.arm} K={a.k} mem={EVAL_MEMORY} ref={a.pin}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
