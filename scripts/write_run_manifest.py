#!/usr/bin/env python3
"""Write run_manifest.json AT LAUNCH, before training starts.

Written at launch and not at completion on purpose: a run that crashes still
has a manifest, and a crashed run's provenance is exactly what you need to
decide whether to count it. See docs/RECORD.md for the full schema and the
literature behind each field.

Three fields are load-bearing beyond the rest, each closing a channel through
which arms could differ with nothing erroring:

    weights_block_sha256   a data change disguised as a label change      (I2)
    gpu_product            a seed pair split across GPU models            (I7b)
    seeds.*                head size perturbing data order via one RNG    (I7a)

Fields that cannot be determined are written as null, never guessed. A guessed
provenance value is worse than a missing one: it looks like evidence.

Run inside the training container, before weaver starts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import socket
import subprocess
import sys
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

STREAMS = ("trunk_init", "head_init", "data_sampling", "dropout")


def _sha256(path: pathlib.Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def _weights_block_sha256(path: pathlib.Path) -> str | None:
    """sha256 from `weights:` to EOF -- the same span I2's CI test compares."""
    if not path.exists():
        return None
    text = path.read_text()
    m = re.search(r"^weights:", text, re.M)
    return hashlib.sha256(text[m.start():].encode()).hexdigest() if m else None


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(("git", *args), cwd=ROOT, capture_output=True,
                              text=True, timeout=10).stdout.strip() or None
    except Exception:
        return None


def _derive_seeds(master: int) -> dict[str, int] | None:
    """Four independent sub-seeds, via the committed derivation.

    Imported rather than reimplemented: a second copy of the derivation that
    drifts from the real one would produce a manifest that documents seeds the
    run did not use.
    """
    try:
        from src.utils.reproducibility import derive_all
        return derive_all(master)
    except Exception:
        return None


def _torch_env() -> dict:
    """GPU and framework provenance. Never guessed -- null if torch is absent.

    The cudnn flags here are a PRE-TRAINING snapshot and are named to say so.
    This script runs BEFORE seed_weaver.py, deliberately: a run that dies must
    still leave a manifest. But seed_weaver sets `cudnn.deterministic = True`
    and `cudnn.benchmark = False` in its own process, after this one has exited.
    An earlier version recorded these as plain `cudnn_deterministic` and so
    reported `false` for every deterministic run -- the exact opposite of the
    truth, in a field `docs/RECORD.md` treats as evidence.

    So: `*_at_manifest_write` is what this process observed (torch defaults),
    and `*_intended` is what seed_weaver will set. The authoritative record of
    the realised value is seed_weaver's own stdout in train.log.
    """
    out = {"torch_version": None, "cuda_version": None, "cudnn_version": None,
           "gpu_device_name": None, "n_gpu": None,
           "cudnn_deterministic_at_manifest_write": None,
           "cudnn_benchmark_at_manifest_write": None,
           "cudnn_deterministic_intended": True,
           "cudnn_benchmark_intended": False}
    try:
        import torch
        out["torch_version"] = torch.__version__
        out["cuda_version"] = torch.version.cuda
        out["cudnn_version"] = (torch.backends.cudnn.version()
                                if torch.backends.cudnn.is_available() else None)
        out["cudnn_deterministic_at_manifest_write"] = bool(
            torch.backends.cudnn.deterministic)
        out["cudnn_benchmark_at_manifest_write"] = bool(
            torch.backends.cudnn.benchmark)
        if torch.cuda.is_available():
            out["n_gpu"] = torch.cuda.device_count()
            out["gpu_device_name"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return out


def _weaver_installed() -> str | None:
    """The weaver actually importable in this image.

    CLAUDE.md and this file previously declared the pin `c97de3c`
    (hqucms/weaver-core, dev/custom_train_eval). The image ships RELEASED
    weaver 0.4.17, and the two are not interchangeable: under `c97de3c`,
    `--fetch-step` is `type=float`, so the `--fetch-step 5` these jobs pass
    becomes `5.0` and raises TypeError in the DataLoader worker on the first
    training batch. The jobs work only because the image does NOT contain the
    commit that was claimed. Record what is installed, not what we believe.
    """
    try:
        import weaver
        return getattr(weaver, "__version__", None)
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--num-classes", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--data-config", required=True)
    ap.add_argument("--samples-per-epoch", type=int, required=True)
    ap.add_argument("--num-epochs", type=int, required=True)
    ap.add_argument("--batch-size", type=int, required=True)
    ap.add_argument("--lambda-mass", type=float, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cfg = pathlib.Path(a.data_config)
    if not cfg.is_absolute():
        cfg = ROOT / cfg

    env = _torch_env()
    manifest = {
        "run_id": a.run_id,
        "launched_utc": datetime.now(timezone.utc).isoformat(),
        "finished_utc": None,
        "status": "launched",

        "provenance": {
            "repo_commit": _git("rev-parse", "HEAD"),
            # Ignore the reweighting sidecar the job copies in before this runs.
            # Counting it made `repo_dirty` true on EVERY run, which makes the
            # field useless precisely when it would matter.
            "repo_dirty": bool(_git("status", "--porcelain",
                                    ":(exclude)configs/arms/*.auto.yaml")),
            "repo_dirty_detail": _git("status", "--porcelain",
                                      ":(exclude)configs/arms/*.auto.yaml"),
            # What is ACTUALLY importable, measured. See _weaver_installed.
            "weaver_version_installed": _weaver_installed(),
            # Expected pins, recorded so a mismatch is visible after the fact.
            # NOT c97de3c: that pin is wrong for this image and would break
            # --fetch-step. Kept as a separate field so the discrepancy between
            # what docs claim and what runs stays visible rather than papered
            # over. Reconcile in CLAUDE.md section 4 before quoting either.
            "weaver_pin_claimed_in_docs": "c97de3c",
            "sophon_commit_expected": "9dd6dd6",
            # Tags move; digests do not. Injected by the job spec if available.
            "image_digest": os.environ.get("IMAGE_DIGEST"),
            "arm_config_sha256": _sha256(cfg),
            "weights_block_sha256": _weights_block_sha256(cfg),
            "labelmap_sha256": _sha256(
                ROOT / "configs" / "labelmaps" / "rung_label_maps.v1.csv"),
            "contraction_tree_sha256": _sha256(
                ROOT / "configs" / "labelmaps" / "contraction_tree.v1.yaml"),
        },

        "vocabulary": {
            "arm": a.arm,
            "num_classes": a.num_classes,
            # The ONLY field that may differ within a seed pair.
            "controlled_variable": "num_classes + label map",
        },

        "randomness": {
            "master_seed": a.seed,
            "seeds": _derive_seeds(a.seed),
            "stream_names": list(STREAMS),
            # Which of those four actually govern anything, measured against
            # seed_weaver.py's real execution order rather than its intent.
            #
            # Runtime order is: trunk_init -> data_sampling -> main() -> [inside
            # model_setup] head_init -> build model -> dropout -> DataLoader
            # draws its base_seed. weaver builds trunk AND head in one
            # model_setup call, so:
            #   trunk_init     overwritten by data_sampling before anything is
            #                  built                                    -> INERT
            #   data_sampling  overwritten by head_init/dropout before the
            #                  DataLoader ever draws                    -> INERT
            #   head_init      seeds ALL weight init, trunk included   -> ACTIVE
            #   dropout        seeds data order AND dropout masks      -> ACTIVE
            #
            # I7's actual guarantee is intact, and that is the part that matters:
            # `dropout` is seeded AFTER model construction, so data order is a
            # function of (master_seed, "dropout") alone and CANNOT be offset by
            # head size. All three arms therefore see the same stream order
            # despite K differing. The four-stream scheme is what is recorded
            # above; this field records what is realised, so the manifest does
            # not document a mechanism the run does not implement.
            "effective_streams": {
                "all_weight_init": "head_init",
                "data_order_and_dropout": "dropout",
                "inert": ["trunk_init", "data_sampling"],
            },
            "cudnn_deterministic_at_manifest_write":
                env["cudnn_deterministic_at_manifest_write"],
            "cudnn_benchmark_at_manifest_write":
                env["cudnn_benchmark_at_manifest_write"],
            "cudnn_deterministic_intended": env["cudnn_deterministic_intended"],
            "cudnn_benchmark_intended": env["cudnn_benchmark_intended"],
        },

        "hardware": {
            # Both strings, from different layers: their agreement is the check
            # that the pod did not move after scheduling.
            "gpu_product_nodelabel": os.environ.get("GPU_PRODUCT"),
            "gpu_device_name": env["gpu_device_name"],
            "n_gpu": env["n_gpu"],
            "node_name": os.environ.get("NODE_NAME"),
            "pod_name": os.environ.get("POD_NAME") or socket.gethostname(),
            "region": os.environ.get("REGION"),
            "torch_version": env["torch_version"],
            "cuda_version": env["cuda_version"],
            "cudnn_version": env["cudnn_version"],
        },

        "data_stream": {
            "data_config_path": str(a.data_config),
            "selection": "200 < jet_pt < 2500 & 20 < jet_sdmass < 500",
            "batch_size": a.batch_size,
            "samples_per_epoch": a.samples_per_epoch,
            "num_epochs": a.num_epochs,
            "examples_seen": a.samples_per_epoch * a.num_epochs,
            # I3. Filled by the G0 smoke run; null until then, never guessed.
            "first_1e6_jet_id_sha256": None,
        },

        "optimization": {
            "optimizer": "ranger",
            "lr_schedule": "flat+decay",
            "amp_enabled": True,
            "lambda_mass": a.lambda_mass,
            "mass_head": a.lambda_mass is not None,
        },

        "compute": {
            "wall_clock_hours": None,
            "gpu_hours": None,
            "throughput_entries_per_s": None,
            "peak_gpu_mem_gb": None,
            # ParT's convention: fvcore counts MACs and labels them FLOPs.
            # Quoting 2*MACs would look like twice ParT's published 340 M on
            # what is essentially ParT's architecture. See docs/RECORD.md 2.1.
            "macs_per_jet_fwd": None,
            "flops_convention": "MACs, as in ParT Table 4 (PMLR 162:18281)",
        },
    }

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n")

    p = manifest["provenance"]
    print(f"manifest -> {out}")
    print(f"  arm={a.arm} K={a.num_classes} seed={a.seed}")
    print(f"  weights_block_sha256={(p['weights_block_sha256'] or 'NONE')[:16]}")
    print(f"  gpu={manifest['hardware']['gpu_device_name']} "
          f"nodelabel={manifest['hardware']['gpu_product_nodelabel']}")
    print(f"  examples_seen={manifest['data_stream']['examples_seen']:,}")
    if p["weights_block_sha256"] is None:
        print("  WARNING: no weights: block found - I2 cannot be checked for this run",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
