"""Jobs of the v2 zero-fill masking control (plan 2026-10-29, A2 and job J5; DECISIONS_PENDING
16(d) approved the zero fill of the top and q/g inputs only beside this control, and the v1
control had no untrained network, so it could not tell removed information from an input
the network never saw).

Per model -- the 12 runs of the four label sets at best70, and the untrained trunks
init-s1..s3 at init -- one GPU job fillctl-<stem>-raunav, on any product but the RTX 3090
(as scripts/build_linprobe_jobs.py's). For each mask in MASKS (the slots
configs/finetune/<mask>.yaml zero-fills, masked on JetClass-II by the configs
scripts/build_downstream_configs.py generates):
  1. experiments/EVAL/extract_v2.py, as the linear-probe extraction, of the training subset
     train_N1000000 and validation through configs/finetune/JetClassII_L162_mask<mask>.yaml
     (162-group labels: linear_probe_v2.py reads the native ones from the subsets' parquet)
     and of the first 2,000,000 TEST2M jets through JetClassII_base_mask<mask>.yaml (native
     labels), into FC_ROOT/<mask>/<model>/jc2/<split>/<checkpoint>/;
  2. experiments/EVAL/linear_probe_v2.py --datasets jc2_188 on them, into FC_ROOT/<mask>/fits.
The unmasked numbers are the linear-probe refit's (LP_ROOT/fits_v3), on the same jets.

    python3 scripts/build_fill_control_v2.py [--pin-not-yet-tagged] [--only mtx-l188-s1 ...] [--check-only]
"""
from __future__ import annotations

import argparse
import importlib.util
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "experiments" / "EVAL" / "k8s" / "v2" / "fillcontrol"
# mtx-s2.18 (4f92021, the linear-probe refit) was tagged before this builder was committed
PIN = "mtx-s2.19"
FC_ROOT = "/data/results/eval/v2_fillcontrol"
MASKS = ("TopReference", "EnergyFlowQG")
N_TRAIN = 1_000_000
HIERARCHY = tuple(f"mtx-{a}-s{s}" for a in ("l188", "l162", "r42q1", "r16q1") for s in (1, 2, 3))
# What the tag must carry for what the job runs.
NEEDED = {"experiments/EVAL/extract_v2.py": "--no-pooled",
          "experiments/EVAL/linear_probe_v2.py": "def native_metrics",
          **{f"configs/finetune/JetClassII_{s}_mask{m}.yaml": "ZERO-FILL CONTROL"
             for m in MASKS for s in ("L162", "base")}}


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


LP = _load("build_linprobe_jobs", "scripts/build_linprobe_jobs.py")
BX, BF = LP.BX, LP.BF


def models() -> list[tuple[str, str, str, int, int, str]]:
    """(model, run directory, rung, classes, regression outputs, checkpoint)."""
    out = [(m, run, rung, k, reg, ckpts[0]) for m, run, rung, k, reg, ckpts in LP.models()
           if m in HIERARCHY or m.startswith("init-")]
    assert len(out) == 15 and all(m.startswith("init-") or c == "best70" for m, *_, c in out), out
    return out


def calls(model: str, ckpt: str) -> list[str]:
    """Each mask's three extract calls and its probe fit."""
    lines = []
    for m in MASKS:
        ft, base = f"configs/finetune/JetClassII_L162_mask{m}.yaml", f"configs/finetune/JetClassII_base_mask{m}.yaml"
        lines += [f"          MASK={m}\n",
                  f"          extract jc2 train_N{N_TRAIN} {ft} 0 {LP.JC2}/train_N{N_TRAIN}_s1.parquet\n",
                  f"          extract jc2 val {ft} 0 {LP.JC2}/val.parquet\n",
                  f"          extract jc2 test {base} 2000000 {BF.test2m_list()}\n",
                  f"          python3 experiments/EVAL/linear_probe_v2.py --root {FC_ROOT}/{m} --model {model} \\\n"
                  f"            --checkpoints {ckpt} --datasets jc2_188 --jc2-parquet {LP.JC2} \\\n"
                  f"            --out {FC_ROOT}/{m}/fits || halt\n"]
    return lines


def planned_bytes() -> int:
    """What every job writes to /data: each mask's rows of the three splits at one checkpoint
    (the linear-probe extraction's row size and test and validation counts)."""
    return len(models()) * len(MASKS) * (N_TRAIN + LP.OTHER_JETS["jc2"]) * LP.ROW_BYTES


def spec(model, run, rung, k, reg, ckpt) -> str:
    body = (LP.EXTRACT_FN.format(root=f"{FC_ROOT}/${{MASK}}", model=model, run_dir=f"{BX.V2_ROOT}/{run}",
                                 rung=rung, k=k, reg=reg, ckpts=ckpt)
            + "".join(calls(model, ckpt))
            + f'          echo "masking control of {model} done"\n')
    # the probe fits are full-batch L-BFGS in torch on the same GPU (linear_probe_v2.Logit)
    return LP.off_grid_gpu(BX.V1ERR_TEMPLATE.format(
        name=f"fillctl-{model.removeprefix('mtx-')}-raunav", image=BX.IMAGE, pin=PIN, body=body,
        mem=LP.X_MEM, cpu="8", gpu_req=', nvidia.com/gpu: "1"', gpu_check=BX.GPU_CHECK.format(),
        node_exclude=BX.NODE_EXCLUDE))


def build(only=None) -> dict[str, str]:
    return {f"job-fillctl-{m[0].removeprefix('mtx-')}-raunav.yaml": spec(*m)
            for m in models() if not only or m[0] in only}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pin-not-yet-tagged", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args(argv)
    # the mask configs must be what their generator writes today
    subprocess.run([sys.executable, str(ROOT / "scripts/build_downstream_configs.py"), "--check-only"],
                   check=True)
    BX.verify_pin(PIN, list(NEEDED), a.pin_not_yet_tagged, NEEDED)
    specs = build(a.only)
    if not a.check_only:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        for name, text in specs.items():
            (OUT_DIR / name).write_text(text)
    print(f"{len(specs)} specs {'checked' if a.check_only else 'written to ' + str(OUT_DIR.relative_to(ROOT))}; "
          f"they write {planned_bytes() / 1e9:.1f} GB to /data")
    return 0


if __name__ == "__main__":
    sys.exit(main())
