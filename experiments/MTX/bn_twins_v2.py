#!/usr/bin/env python3
"""BatchNorm-recomputed twins of a finished v2 run's reported checkpoints.

PRESPEC A14's BatchNorm rule fired (outcome 2026-10-03, experiments/FIGS/data/
head_epoch_diag/head_bn_diag.json: recomputing BatchNorm alone repairs 21 of the 22
defective stored v1 epochs). So every reported v2 checkpoint also gets a model with its
BatchNorm statistics recomputed on the same seeded epoch-80 stream the weight average
used, reported beside it, for the frozen readouts only. The reported checkpoints that
keep their stored statistics are best70 (best_window_epoch.json) and the global best
(best_epoch.json); the weight average has its statistics recomputed already.

pretrain_v2 at the grid's tag does not write these models, so this entry point writes
them after the run, with pretrain_v2's own functions, as finalize_v2 redoes the end
step: the run's seeds and configuration from recipe.json, the trimmer counters of its
last resume file, and the GPU product it trained on (the weight average's BatchNorm
pass ran there). For each distinct epoch it writes

  net_epoch-<e>_bn_state.pt   the epoch's state file, BatchNorm statistics recomputed
  net_epoch-<e>_bn.json       the input's and the output's sha256, the BatchNorm sample,
                              the tags it serves (best70, bestval) and the code

and writes nothing unless the BatchNorm sample is the weight average's own: the same
number of jets and the same rows in the same order (rows_sha256 as recorded in
net_wavg<F>-<L>.json). A twin already written is left alone.

It refuses (EXIT_HALT) a run that is not DONE, one without its weight average's record,
a state file or the last epoch's resume file, one whose training files on disk are not
the ones the recipe counted, and a GPU product other than the one the run trained on.

Run it as the training job runs pretrain_v2, from the repository root with the arm's
reweighting sidecar beside its config:
  python3 experiments/MTX/bn_twins_v2.py --out /data/results/mtx_v2/RUN
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

DRIVER = "experiments/MTX/bn_twins_v2.py"
TAGS = {"best70": "best_window_epoch.json", "bestval": "best_epoch.json"}


def twin_paths(out: pathlib.Path, epoch: int) -> tuple[pathlib.Path, pathlib.Path]:
    return out / f"net_epoch-{epoch}_bn_state.pt", out / f"net_epoch-{epoch}_bn.json"


def tag_epochs(out: pathlib.Path) -> dict[int, list[str]]:
    """{epoch: the reported tags it is} from the run's own records."""
    by = {}
    for tag, rec in TAGS.items():
        by.setdefault(int(json.loads((out / rec).read_text())["epoch"]), []).append(tag)
    return by


def run_args(out: pathlib.Path) -> tuple[dict, argparse.Namespace]:
    """(recipe.json, the run's arguments) as finalize_v2 reads them."""
    rec = json.loads((out / "recipe.json").read_text())
    return rec, argparse.Namespace(**{k: rec[k] for k in pv.RECIPE_ARGS}, data_train=rec["data_train"],
                                   out=str(out))


def build(out: pathlib.Path, a, dev):
    """(model, seeds, sidecar, data config): the run's model with the trimmer counters of
    its last resume file, so the BatchNorm pass trims as the trained model did."""
    import torch

    import stream_v2 as sv
    from src.utils.reproducibility import derive_all
    pv.set_numerics(a)
    seeds = derive_all(a.seed)
    side = pv.sidecar(a.data_config)
    data_config = sv.load_config(side, a.extra_selection)
    model, _ = pv.make_model(a, pv.objective_kind(a), data_config, seeds, dev)
    st = torch.load(out / f"net_epoch-{a.num_epochs - 1}_resume.pt", map_location="cpu", weights_only=False)
    pv.set_trimmer_counters(model, st["trimmer_counters"])
    return model, seeds, side, data_config


def bn_recompute(a, model, ds_train, seeds, dev, amp: bool, loader_kw, input_names) -> dict:
    """The weight average's BatchNorm pass (pretrain_v2.write_weight_average) on whatever
    weights `model` holds: the same seed, the epoch-80 stream, BN_JETS jets, recorded."""
    from torch.utils.data import DataLoader
    pv.seed_all(pv.epoch_seed(seeds["dropout"], "bn-recompute", 0))
    ds_train.set_epoch(a.num_epochs)
    rec = pv.StreamRecord()

    def recorded(it):
        seen = 0
        for X, y, Z in it:
            take = min(len(Z["_rowid"]), pv.BN_JETS - seen)
            rec.update({k: v[:take] for k, v in Z.items()})
            seen += take
            yield X, y, Z
    it = iter(DataLoader(ds_train, **loader_kw))
    n_bn = pv.recompute_bn(model, recorded(it), pv.BN_JETS, dev, amp, input_names)
    del it
    names = {v: k for k, v in ds_train.file_index.items()}
    return {"n_jets": int(rec.n), "batchnorm_layers": n_bn, "stream_epoch": a.num_epochs,
            "seed_data": seeds["data_sampling"], "rows_sha256": rec.h.hexdigest(),
            "files": sorted(names[i] for i in rec.files),
            "dropout_seed": pv.epoch_seed(seeds["dropout"], "bn-recompute", 0),
            "mode": "train, no gradient, cumulative average (momentum None)"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="run directory")
    out = pathlib.Path(ap.parse_args(argv).out)
    if not (out / "DONE").exists():
        print(f"FATAL: {out} is not DONE; its reported checkpoints are not final", flush=True)
        return pv.EXIT_HALT
    rec, a = run_args(out)
    window = pv.last_epochs(a.num_epochs)
    wavg_json = out / f"net_wavg{window[0]}-{window[-1]}.json"
    last = a.num_epochs - 1
    by_epoch = tag_epochs(out)
    missing = [str(p) for p in [wavg_json, out / f"net_epoch-{last}_resume.pt",
                                *(out / f"net_epoch-{e}_state.pt" for e in by_epoch)] if not p.exists()]
    if missing:
        print(f"FATAL: {out} lacks {missing}", flush=True)
        return pv.EXIT_HALT
    todo = sorted(e for e in by_epoch if not twin_paths(out, e)[1].exists())
    if not todo:
        print(f"[bn_twins_v2] {out.name}: every twin is written ({sorted(by_epoch)})", flush=True)
        return 0
    want = json.loads(wavg_json.read_text())["bn_recompute"]
    counts = {k: len(v) for k, v in pv.to_file_dict(a.data_train).items()}
    if counts != rec["data_train_n"]:
        print(f"FATAL: training files on disk per family {counts}; the run read {rec['data_train_n']}", flush=True)
        return pv.EXIT_HALT
    if a.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    dev = torch.device("cpu" if rec["device"] == "cpu" else "cuda")
    if dev.type == "cuda":
        why = pv.gpu_missing("cuda")
        if why:
            print(f"FATAL: the run trained on {rec['device']} and there is no usable GPU ({why})", flush=True)
            return pv.EXIT_NO_GPU
        if torch.cuda.get_device_name(0) != rec["device"]:
            print(f"FATAL: the run trained on {rec['device']}, this is {torch.cuda.get_device_name(0)}", flush=True)
            return pv.EXIT_HALT
    model, seeds, side, data_config = build(out, a, dev)
    code = {**pv.code_version(), "driver": DRIVER}
    amp = a.use_amp and dev.type == "cuda"
    for e in todo:
        src = out / f"net_epoch-{e}_state.pt"
        model.load_state_dict(torch.load(src, map_location="cpu", weights_only=True))
        bn = bn_recompute(a, model, pv.train_stream(a, side, seeds), seeds, dev, amp,
                          pv.loader_kwargs(a, dev), list(data_config.input_names))
        if (bn["n_jets"], bn["rows_sha256"]) != (want["n_jets"], want["rows_sha256"]):
            print(f"FATAL: epoch {e}'s BatchNorm sample ({bn['n_jets']} jets, rows {bn['rows_sha256'][:12]}) "
                  f"is not the weight average's ({want['n_jets']}, {want['rows_sha256'][:12]})", flush=True)
            return pv.EXIT_HALT
        state, meta = twin_paths(out, e)
        pv.torch_save(model.state_dict(), state)
        pv.write_json(meta, {"epoch": e, "tags": by_epoch[e], "inputs": {str(e): pv.sha256_file(src)},
                             "sha256": pv.sha256_file(state), "bn_recompute": bn,
                             "device": rec["device"], "code": code})
        print(f"[bn_twins_v2] {out.name}: {state.name} ({'+'.join(by_epoch[e])}), "
              f"{bn['batchnorm_layers']} BatchNorm layers recomputed on {bn['n_jets']} jets, "
              f"the weight average's sample", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
