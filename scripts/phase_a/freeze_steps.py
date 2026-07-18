#!/usr/bin/env python3
"""Phase A §6-A.5(d): step freeze for the E2 base config (G0 artifact).

Records N_steps_full, batch size, jets/epoch, and optimizer/schedule
hyperparameters from the frozen E2 base config, plus SHA256 hashes of the
files that define it. The combined hash is the `config_hash` referenced by
PLAN.md — every E2 run's provenance must cite it.

Run from the repo root:  python3 scripts/phase_a/freeze_steps.py
Writes: registry/gates/step_freeze.json
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FROZEN_FILES = [  # the files that DEFINE the recipe, in hash order
    "experiments/E2/train_e2.sh",
    "experiments/E2/data/JetClassII_full_base.yaml",
    # added 2026-07-19 (amendment): the network/architecture file is
    # recipe-critical and belongs in the freeze — gap exposed when its
    # weaver-0.4.17-incompatible forward crashed model_setup.
    "experiments/E2/networks/example_ParticleTransformer_sophon.py",
]


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    hashes = {f: sha256(REPO / f) for f in FROZEN_FILES}
    combined = hashlib.sha256(
        "".join(hashes[f] for f in FROZEN_FILES).encode()).hexdigest()

    # cross-check the recipe numbers actually in train_e2.sh (fail loud on drift)
    sh = (REPO / "experiments/E2/train_e2.sh").read_text()
    assert "--batch-size 512" in sh and "--start-lr 5e-4" in sh, "recipe drift: batch/lr"
    assert "--optimizer ranger" in sh, "recipe drift: optimizer"
    epochs = int(re.search(r"epochs=(\d+)", sh).group(1)) if re.search(r"epochs=(\d+)", sh) else 80
    spe = re.search(r"samples_per_epoch=\$\(\((.+?)\)\)", sh)

    batch = 512
    jets_per_epoch = 10_240_000              # 10000 * 1024 (Sophon recipe)
    n_epochs = 80
    n_steps_full = n_epochs * jets_per_epoch // batch
    assert n_steps_full == 1_600_000, n_steps_full

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip()
    except Exception:
        commit = None

    out = {
        "artifact": "PLAN v5.1 §6-A.5(d) step freeze  ⟨BIND:G0⟩",
        "frozen_on": str(date.today()),
        "upstream_pin": "jet-universe/sophon@9dd6dd6 (train_sophon.sh recipe)",
        "N_steps_full": n_steps_full,
        "batch_size": batch,
        "jets_per_epoch": jets_per_epoch,
        "num_epochs": n_epochs,
        "epoch_definition": "PLAN 'epoch' == N_steps_full/80 == 20,000 steps",
        "optimizer": {"name": "ranger (weaver)", "start_lr": 5e-4,
                      "schedule": "weaver default flat+decay", "amp": True},
        "head": "fc_params [(512, 0.1)]",
        "loader": {"num_workers": 5, "fetch_step": 1.0, "data_split_num": 200,
                   "single_gpu_only": "seed_weaver.py wraps weaver.train in-process"},
        "train_e2_sh_epochs_var": epochs,
        "samples_per_epoch_expr": spe.group(1) if spe else None,
        "file_hashes_sha256": hashes,
        "config_hash": combined,
        "git_commit_at_freeze": commit,
        "note": ("Stage-0 LR grid per §9.2 = {5e-4, 2.5e-4}. Any change to the "
                 "frozen files invalidates config_hash and re-opens G0."),
    }
    dest = REPO / "registry/gates/step_freeze.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {dest}\nconfig_hash = {combined}")


if __name__ == "__main__":
    main()
