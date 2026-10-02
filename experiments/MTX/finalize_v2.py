#!/usr/bin/env python3
"""Finish a v2 pretraining run whose epochs all completed but whose end step failed.

pretrain_v2 ends a run with best_window_epoch.json, the weight average of the last ten
epochs (net_wavg<F>-<L>_state.pt and .json) and DONE. If that step fails after the last
epoch, pretrain_v2 cannot simply be rerun under fixed code: its recipe check refuses a
different code version. This entry point redoes the step with pretrain_v2's own
functions, the run's seeds and configuration from its recipe.json and the trimmer
counters of its last resume file, without the code-version check, and records its own
code version in what it writes.

It refuses (EXIT_HALT) a run that is DONE, one missing a state file of the last ten
epochs or the last epoch's resume file, one whose training files on disk are not the
ones the recipe counted, and a GPU product other than the one the run trained on.

Run it as the training job runs pretrain_v2, from the repository root with the arm's
reweighting sidecar beside its config:
  python3 experiments/MTX/finalize_v2.py --out /data/results/mtx_v2/RUN
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE))

import pretrain_v2 as pv  # noqa: E402

DRIVER = "experiments/MTX/finalize_v2.py"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="run directory")
    out = pathlib.Path(ap.parse_args(argv).out)
    if (out / "DONE").exists():
        print(f"FATAL: {out} is DONE; nothing to finish", flush=True)
        return pv.EXIT_HALT
    rec = json.loads((out / "recipe.json").read_text())
    a = argparse.Namespace(**{k: rec[k] for k in pv.RECIPE_ARGS}, data_train=rec["data_train"], out=str(out))
    last = a.num_epochs - 1
    missing = [str(p) for p in [out / f"net_epoch-{e}_state.pt" for e in pv.last_epochs(a.num_epochs)]
               + [out / f"net_epoch-{last}_resume.pt"] if not p.exists()]
    if missing:
        print(f"FATAL: {out} is not a run whose epochs all finished: no {missing}", flush=True)
        return pv.EXIT_HALT
    counts = {k: len(v) for k, v in pv.to_file_dict(a.data_train).items()}
    if counts != rec["data_train_n"]:
        print(f"FATAL: training files on disk per family {counts}; the run read {rec['data_train_n']}", flush=True)
        return pv.EXIT_HALT
    if a.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    import stream_v2 as sv
    from src.utils.reproducibility import derive_all

    pv.set_numerics(a)
    dev = torch.device("cpu" if rec["device"] == "cpu" else "cuda")
    if dev.type == "cuda":
        why = pv.gpu_missing("cuda")
        if why:
            print(f"FATAL: the run trained on {rec['device']} and there is no usable GPU ({why})", flush=True)
            return pv.EXIT_NO_GPU
        if torch.cuda.get_device_name(0) != rec["device"]:
            print(f"FATAL: the run trained on {rec['device']}, this is {torch.cuda.get_device_name(0)}", flush=True)
            return pv.EXIT_HALT
    kind = pv.objective_kind(a)
    seeds = derive_all(a.seed)
    side = pv.sidecar(a.data_config)
    data_config = sv.load_config(side, a.extra_selection)
    model, _ = pv.make_model(a, kind, data_config, seeds, dev)
    # the BatchNorm pass trims as the trained model did: its counters, not a fresh model's
    st = torch.load(out / f"net_epoch-{last}_resume.pt", map_location="cpu", weights_only=False)
    pv.set_trimmer_counters(model, st["trimmer_counters"])
    code = {**pv.code_version(), "driver": DRIVER}
    window = pv.write_window_best(out, a.num_epochs, code)
    pv.write_weight_average(out, a, model, pv.train_stream(a, side, seeds), seeds, dev,
                            a.use_amp and dev.type == "cuda", pv.loader_kwargs(a, dev),
                            list(data_config.input_names), code)
    best = json.loads((out / "best_epoch.json").read_text())
    (out / "DONE").write_text(json.dumps({**best, "finalized_by": code}) + "\n")
    print(f"[finalize_v2] {out.name}: window best epoch {window['epoch']}, weight average written, DONE",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
