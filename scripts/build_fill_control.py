#!/usr/bin/env python3
"""Emit the zero-fill control job (docs/PRD_PLAN.md 4.5, DECISIONS_PENDING 16d).

WHAT THE CONTROL IS FOR. The top and q/g rows zero-fill features those datasets
do not carry -- ten slots for top, four for q/g. Item 16(d) approves that fill
ONLY alongside a control that masks the SAME slots on JetClass-II, where the
unmasked number also exists, and shows the arm ORDERING survives. Without it a
granularity effect and an imputation artefact are indistinguishable.

WHY THIS IS A GENERATOR AND NOT A HAND-WRITTEN SPEC. The extraction must read
the same 335 test files IN THE SAME ORDER as the baseline it is compared
against, and that list is NOT reconstructible by brace expansion: it is
interleaved across families (Res2P_0250, Res34P_1075, QCD_0350, ...), which is
why all 188 classes appear within the first 400,000 rows. The list is therefore
copied verbatim out of the committed extraction spec.

WHY 400,000. Re-extracting at 2 M would cost 5x the GPU for statistics the
control does not need: the primary b-vs-c resonant probe already has 12,866 vs
13,218 jets in the first 400,000 (measured), and the QCD b-vs-c task 1,584 vs
3,458. All three legs of an arm -- unmasked and both masks -- are extracted
here, at the same checkpoint over the same file list with the same --max-jets,
so they are the same jets by construction.

THAT ALIGNMENT IS CHECKED EXPLICITLY, and this file used to claim a check that
did not exist. probe.py's label188 sha256 refusal only fires between the arms
passed to ONE invocation, and the control runs unmasked and each mask as
SEPARATE invocations -- so nothing ever compared a masked leg against its own
baseline. The emitted script now diffs the three legs' label188 sha256 per arm
before any probe runs.

Run:  python3 scripts/build_fill_control.py [--pin TAG]
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments" / "EVAL" / "k8s" / "job-eval-fillcontrol-raunav.yaml"
EXTRACT_SPEC = ROOT / "experiments" / "EVAL" / "k8s" / "job-extract-mtx-r16q1-s2-raunav.yaml"
IMAGE = "gitlab-registry.nrp-nautilus.io/escheuller/transfer-learning:cu121"
PIN = "mtx-s1.22"
N_JETS = 400_000

# arm : checkpoint : K. Only arms with BOTH a finished checkpoint and a
# committed downstream baseline can enter; r16q1-s1 has the checkpoint but no
# baseline, so it is deliberately absent.
ARMS = [
    ("r16q1-s2", "/data/results/mtx/mtx-r16q1-s2/net_epoch-79_state.pt", 17),
    ("r16q1-s3", "/data/results/mtx/mtx-r16q1-s3/net_epoch-79_state.pt", 17),
    ("r16q1-s4", "/data/results/mtx/mtx-r16q1-s4/net_epoch-79_state.pt", 17),
    ("l162-s1b", "/data/results/mtx/mtx-l162-s1b/net_epoch-79_state.pt", 162),
]
MASKS = ["TopReference", "EnergyFlowQG"]


def test_list() -> str:
    text = EXTRACT_SPEC.read_text()
    m = re.search(r"--data-test ((?:/jc2/jet_data/\S+\.parquet ?)+)", text)
    if not m:
        sys.exit(f"FATAL: no --data-test list in {EXTRACT_SPEC}")
    files = m.group(1).split()
    if len(files) != 335:
        sys.exit(f"FATAL: expected 335 test files, found {len(files)}")
    return " ".join(files)


HEADER = """  # THE ZERO-FILL CONTROL -- docs/PRD_PLAN.md 4.5, DECISIONS_PENDING item 16(d).
  #
  # Masks on JetClass-II EXACTLY the slots the top (10) and q/g (4) configs must
  # zero-fill, re-extracts frozen features per arm, and re-runs the linear AND
  # MLP probes so the paper can report the penalty and show the arm ORDERING is
  # unchanged under masking. Item 16(d) approves the fill only WITH this.
  #
  # The extraction config is configs/finetune/JetClassII_base_mask*.yaml, NOT the
  # L162 control: extract_features.py runs with configs/data/JetClassII_base.yaml,
  # whose labels: block is the native 188-way jet_label, and the probes key on
  # native labels (label_X_bb 0, label_X_cc 1, label_QCD_bb 169). Extracting
  # through the L162 control would hand them 162-GROUP labels under the same
  # array name: every probe would still run and silently measure another task.
  #
  # 400,000 jets: the primary b-vs-c resonant probe already has 12,866 vs 13,218
  # jets there (measured). Every leg is extracted here at the same checkpoint
  # over the same file list, and the script diffs the three legs' label188
  # sha256 per arm before probing -- probe.py's own sha refusal cannot do it,
  # because it only compares arms WITHIN one invocation and the legs are
  # separate invocations.
  #
  # GPU: 4 arms x 2 masks x 400k forward passes, then CPU probes in the same pod.
"""


def script(pin: str) -> str:
    lines = [
        "          set -euo pipefail",
        f'          git clone --depth 1 --branch "{pin}" \\',
        "            https://github.com/raunavm/transferlearningsophon.git \\",
        "            /workspace/transferlearningsophon",
        "          cd /workspace/transferlearningsophon",
        "          git rev-parse HEAD",
        "          pip install --no-cache-dir -q pyarrow || exit 1",
        "          export PYTHONUNBUFFERED=1",
        "",
        f'          TEST="{test_list()}"',
        "          ROOT_OUT=/data/results/eval/fillcontrol",
        f"          N={N_JETS}",
        "          mkdir -p ${ROOT_OUT}",
        "",
        "          # Every checkpoint must exist before a single GPU minute is",
        "          # spent. The baseline is no longer a pre-existing cache: both",
        "          # legs are extracted here, from this checkpoint.",
        "          for c in " + " ".join(c for _, c, _ in ARMS) + "; do",
        '            [ -f "${c}" ] || { echo "FATAL: no ${c}"; exit 1; }',
        "          done",
        "",
    ]
    for arm, ckpt, k in ARMS:
        lines += [
            f"          # ---- {arm} (K={k})",
            f'          [ -f "{ckpt}" ] || {{ echo "FATAL: no {ckpt}"; exit 1; }}',
            f"          U={{ROOT_OUT}}/{arm}/unmasked".replace("{ROOT_OUT}", "${ROOT_OUT}"),
            # The unmasked leg is EXTRACTED at the same checkpoint as the
            # masked legs, not truncated out of features_v2. features_v2 was
            # written from net_best_epoch_state.pt (epochs 74/64/76/78 for
            # r16q1-s2/s3/s4/l162-s1b); the masked legs load epoch 79. A
            # control whose two legs come from different checkpoints measures
            # the zero-fill penalty PLUS a checkpoint change, per arm, with
            # nothing erroring. Epoch 79 is the side that moves, because
            # DECISIONS_PENDING item 18 resolved to epoch 79 uniformly.
            f"          [ -f ${{U}}/label188.npy ] || python3 experiments/EVAL/extract_features.py \\",
            f"            --checkpoint {ckpt} --num-classes {k} --arm {arm}_unmasked \\",
            f"            --data-config configs/data/JetClassII_base.yaml \\",
            f"            --data-test ${{TEST}} --out ${{U}} \\",
            f"            --batch-size 512 --num-workers 1 --fetch-step 1 --max-jets ${{N}}",
        ]
        for mask in MASKS:
            d = f"${{ROOT_OUT}}/{arm}/{mask}"
            lines += [
                f"          [ -f {d}/label188.npy ] || python3 experiments/EVAL/extract_features.py \\",
                f"            --checkpoint {ckpt} --num-classes {k} --arm {arm}_{mask} \\",
                f"            --data-config configs/finetune/JetClassII_base_mask{mask}.yaml \\",
                f"            --data-test ${{TEST}} --out {d} \\",
                f"            --batch-size 512 --num-workers 1 --fetch-step 1 --max-jets ${{N}}",
            ]
        lines.append("")
    lines += [
        '          echo "===== row alignment: label188 sha256 per arm ====="',
        "          for a in " + " ".join(a for a, _, _ in ARMS) + "; do",
        "            ref=",
        "            for leg in unmasked " + " ".join(MASKS) + "; do",
        "              h=$(python3 -c \"import hashlib,numpy,sys;"
        "print(hashlib.sha256(numpy.load(sys.argv[1]).tobytes()).hexdigest())\" \\",
        "                    ${ROOT_OUT}/${a}/${leg}/label188.npy)",
        '              echo "  ${a}/${leg} ${h}"',
        '              if [ -z "${ref}" ]; then ref="${h}"',
        '              elif [ "${h}" != "${ref}" ]; then',
        '                echo "FATAL: ${a}/${leg} labels differ from its own baseline;"',
        '                echo "       the legs are not the same jets, so the fill"',
        '                echo "       penalty would be a different-sample effect."; exit 1',
        "              fi",
        "            done",
        "          done",
        "",
    ]
    for tag, sub in [("unmasked", "unmasked")] + [(m, m) for m in MASKS]:
        feats = " ".join(f"{a}=${{ROOT_OUT}}/{a}/{sub}" for a, _, _ in ARMS)
        lines += [
            f'          echo "===== probes: {tag} ====="',
            f"          python3 experiments/EVAL/probe.py \\",
            f"            --features {feats} \\",
            f"            --out ${{ROOT_OUT}}/probe_{tag} --bootstrap 2000",
            "",
        ]
    lines += [
        '          echo "===== summary ====="',
        "          for t in unmasked " + " ".join(MASKS) + "; do",
        '            echo "--- ${t}"; cat ${ROOT_OUT}/probe_${t}/*.json',
        "          done",
        '          echo "FILL CONTROL COMPLETE"',
    ]
    return "\n".join(lines) + "\n"


def build(pin: str) -> str:
    return (
        "apiVersion: batch/v1\nkind: Job\nmetadata:\n"
        "  # GENERATED by scripts/build_fill_control.py -- do not hand-edit. Regenerate.\n"
        + HEADER
        + "  name: eval-fillcontrol-raunav\n  namespace: cms-ml\nspec:\n"
        "  backoffLimit: 6\n  template:\n    spec:\n      restartPolicy: Never\n"
        "      containers:\n      - name: fillcontrol\n"
        f"        image: {IMAGE}\n"
        '        command: ["/bin/bash", "-c"]\n        args:\n        - |\n'
        + script(pin)
        + "        volumeMounts:\n        - { name: data, mountPath: /data }\n"
        "        - { name: jc2, mountPath: /jc2, readOnly: true }\n"
        "        resources:\n"
        '          requests: { memory: "32Gi", cpu: "4", nvidia.com/gpu: "1", ephemeral-storage: "20Gi" }\n'
        '          limits:   { memory: "32Gi", cpu: "4", nvidia.com/gpu: "1", ephemeral-storage: "20Gi" }\n'
        "      affinity:\n        nodeAffinity:\n"
        "          requiredDuringSchedulingIgnoredDuringExecution:\n"
        "            nodeSelectorTerms:\n            - matchExpressions:\n"
        "              - key: topology.kubernetes.io/region\n"
        '                operator: In\n                values: ["us-west"]\n'
        "              - key: kubernetes.io/hostname\n"
        '                operator: NotIn\n'
        '                values: ["ry-gpu-03.sdsc.optiputer.net", "nautilus-ext-gpu01.fullerton.edu", "hcc-chase-shor-c4705.unl.edu"]\n'
        "      volumes:\n      - name: data\n        persistentVolumeClaim:\n"
        "          claimName: transfer-learning-vol\n"
        "      - name: jc2\n        persistentVolumeClaim:\n"
        "          claimName: tn-pvc-base-jetclass2\n          readOnly: true\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pin", default=PIN)
    a = ap.parse_args()
    text = build(a.pin)
    d = yaml.safe_load(text)
    assert d["metadata"]["name"].endswith("-raunav")
    args = d["spec"]["template"]["spec"]["containers"][0]["args"][0]
    for tok in ("TopReference", "EnergyFlowQG", "probe.py"):
        assert tok in args, f"{tok} missing from the emitted script"
    # Every extract_features call -- masked and unmasked alike -- must load the
    # SAME checkpoint, or the control measures a checkpoint change too.
    ckpts = set(re.findall(r"--checkpoint (\S+)", args))
    for arm, ckpt, _ in ARMS:
        assert ckpt in ckpts, f"{arm}: {ckpt} never extracted"
    assert ckpts == {c for _, c, _ in ARMS}, f"unexpected checkpoints: {ckpts}"
    assert "truncate_features.py" not in args, (
        "the unmasked leg must be extracted at the masked legs' checkpoint, "
        "not truncated out of a best-epoch cache")
    OUT.write_text(text)
    print(f"{OUT.name} written (pin {a.pin}, {len(ARMS)} arms x {len(MASKS)} masks, N={N_JETS:,})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
