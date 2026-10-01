#!/usr/bin/env python3
"""Does this pod's GPU answer? Exit 0 if a tensor can be made on it and the device
synchronises, 1 otherwise (retry logic of 2026-10-01, scripts/build_ft_jobs.py).

Run at the start of every resumable fine-tuning pod -- a node whose GPU has
fallen over is then refused before any cell is touched -- and by the job's EXIT
trap after a failed step: if the GPU no longer answers, the failure is the
node's (NODE_FAULT), not the cell's. On 2026-09-30 ry-gpu-10 failed two read-outs
of one cell this way ("CUDA error: unknown error", then "Cannot access
accelerator device when none is available") and they were counted as the cell's
two real failures.
"""
import sys


def main() -> int:
    try:
        import torch
        x = torch.zeros(1).cuda()
        torch.cuda.synchronize()
        print(f"gpu_probe: {torch.cuda.get_device_name(x.device)} answers", flush=True)
        return 0
    except Exception as exc:                # noqa: BLE001 -- any failure is the answer
        print(f"gpu_probe: the GPU does not answer: {type(exc).__name__}: {exc}", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
