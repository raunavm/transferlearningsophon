#!/usr/bin/env python3
"""BatchNorm statistics recomputed on the v1 output-layer diagnostic's checkpoints.

WHY (docs/PRESPEC_2026-09.md, amendment A14, analysis rule "Checkpoint robustness"):
before any v2 number is read, the defective v1 epochs are rescored with BatchNorm
statistics recomputed and no averaging, to separate the two parts of the
weight-average model. experiments/DIAG/head_epoch_diag.py found the weight average of
epochs 70-79 sound where single epochs were defective. Its average takes the mean of
every floating tensor, the BatchNorm running statistics included, and recomputes
nothing. The v2 robustness model (experiments/MTX/pretrain_v2.py write_weight_average)
averages the weights and then recomputes the BatchNorm statistics on 200,000 training
jets. ParT's six BatchNorm layers sit in the particle and pair embeddings, upstream of
every dropout. Two questions: does recomputing the BatchNorm statistics alone, with no
averaging, repair a defective output layer? And is the v2-style model (average, then
recompute) as sound as the buffer-averaged one?

WHAT. For each run, from ONE job on ONE GPU, on the 500,000 test jets of the v1
diagnostic (same arguments; the native-label sha256 must equal the committed record):
  e70..e79        the stored epochs, re-scored here and compared with the committed v1
                  values (a reproducibility check);
  e70_bn..e79_bn  the same states with the BatchNorm statistics recomputed, nothing averaged;
  wavg_buf        the mean of e70..e79 with the BatchNorm buffers averaged too
                  (head_epoch_diag.average_states: the v1 diagnostic's weight average);
  wavg_bn         the mean of e70..e79 (pretrain_v2.average_states), BatchNorm statistics
                  then recomputed (the v2 robustness model);
  best, best_bn   the best-validation checkpoint, when it is not one of e70..e79.
Every state is written to disk and scored as head_epoch_diag scores a checkpoint: the
same model build, loading guards, hook and per-jet arrays. The derived states go to
--state-dir, which the job puts on the pod's local disk: they are not kept (about 0.9 GB
over the eight runs), and each one's sha256 is in its run's DONE. Only the per-jet arrays
and the records go to --out. It runs on one GPU and refuses to start without one
(EXIT_NO_GPU); --allow-cpu, for tests, runs on the CPU, without autocast.

THE BATCHNORM SAMPLE. pretrain_v2.BN_JETS (200,000) training jets, drawn ONCE per job by
the v2 training stream (experiments/MTX/stream_v2.py in train mode: the nominal
reweighted class mix) over the training files the v1 runs trained on, with the
reweighting of the shared weights block, and one recorded seed. It is held in memory and
used for every checkpoint of every run, so every comparison is paired. The recompute is
pretrain_v2.recompute_bn itself (reset, momentum None, train mode, no gradient, the
training autocast), with the torch, numpy and python generators seeded the same way
before each checkpoint and SequenceTrimmer past its warm-up, as at the end of training.
Every recomputed state is checked against its input: the same keys, every parameter
bitwise equal, and only BatchNorm running_mean, running_var and num_batches_tracked changed.

Usage (the job spec is experiments/DIAG/k8s/job-diag-head-bn-raunav.yaml):
    head_bn_diag.py infer   --runs RUN_DIR:RUNG:K:NUM_REG ... --data-test ... --max-jets N --stride S
                            --align-with CACHE --bn-config ARM.yaml --bn-train FAMILY:PATH ...
                            --state-dir LOCAL_DIR --out DIR
    head_bn_diag.py analyse --out DIR --json FILE
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import sys
import time

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


HD = _load("head_epoch_diag", "experiments/DIAG/head_epoch_diag.py")
PV = _load("pretrain_v2", "experiments/MTX/pretrain_v2.py")

COMMITTED = REPO / "experiments/FIGS/data/head_epoch_diag/head_epoch_diag_v2.json"
BN_SEED = 20261001        # data_sampling seed of the BatchNorm sample: any fixed value, recorded
# The v2 training stream's settings (scripts/build_mtx_launch.py v2_job), and the epoch
# whose stream pretrain_v2.write_weight_average draws its BatchNorm jets from.
BN_STREAM = {"epoch": 80, "batch_size": 512, "split_num": 200, "fetch_step": 1.0, "data_fraction": 0.2}
BN_WORKERS = 5
DIFF_BOOT = 1000
EXIT_NO_GPU = 4           # no usable GPU: the node's fault; the spec maps it to 43, retried uncounted

DEFINITIONS = {
    "eNN": "epoch NN's state file as v1 training stored it",
    "eNN_bn": "the same state with the BatchNorm running statistics recomputed on the BatchNorm sample; "
              "nothing averaged",
    "wavg_buf": "mean of the stored epochs, BatchNorm buffers averaged with the weights "
                "(the v1 diagnostic's 'wavg', head_epoch_diag.average_states)",
    "wavg_bn": "mean of the stored epochs (pretrain_v2.average_states), BatchNorm statistics then "
               "recomputed on the BatchNorm sample (the v2 robustness model)",
    "best": "the best-validation state, scored when it is not one of the stored epochs; best_bn likewise",
    "reference": "each rule's reference is built from the run's stored epochs alone, as in head_epoch_diag",
    "repaired_by_bn": "defective as stored, sound after the BatchNorm recompute alone",
    "sound_made_defective_by_bn": "sound as stored, defective after the BatchNorm recompute alone",
    "paired_bootstrap": "each replicate resamples the test jets once and scores both models on them",
}


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ---------------------------------------------------------------- BatchNorm recompute
def past_trimmer_warmup(model) -> None:
    """SequenceTrimmer as at the end of training: past its warm-up, so every batch is
    trimmed (at random, in train mode). A freshly built model starts at 0 and would feed
    its first batches untrimmed, padding included, into the input BatchNorm. The counter
    is module state outside the state dict: an int in weaver 0.4.17 (warm-up fixed at
    5), a tensor buffer in the dev branch (warmup_steps)."""
    import torch
    for m in model.modules():
        if type(m).__name__ == "SequenceTrimmer":
            n = getattr(m, "warmup_steps", 5)
            if torch.is_tensor(m._counter):
                m._counter.fill_(n)
            else:
                m._counter = n


def bn_buffer_check(model, before: dict, after: dict) -> dict:
    """What a BatchNorm recompute changed. ok: the same keys, every parameter bitwise
    equal, and nothing changed but BatchNorm running_mean, running_var and
    num_batches_tracked."""
    import torch
    bn = {f"{n}.{b}" for n, m in model.named_modules()
          if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
          for b in ("running_mean", "running_var", "num_batches_tracked")}
    params = [n for n, _ in model.named_parameters()]
    same_keys = before.keys() == after.keys()
    changed = sorted(k for k in after if k not in before or not torch.equal(before[k].cpu(), after[k].cpu()))
    params_equal = all(k in before and k in after and torch.equal(before[k].cpu(), after[k].cpu()) for k in params)
    return {"ok": bool(same_keys and params_equal and set(changed) <= bn),
            "same_keys": same_keys, "parameters": len(params), "parameters_bitwise_equal": params_equal,
            "only_batchnorm_buffers_changed": set(changed) <= bn, "batchnorm_layers": len(bn) // 3,
            "running_stats_changed": sum(k in changed for k in bn if not k.endswith("num_batches_tracked")),
            "changed": changed}


def recompute_state(model, state: dict, sample: dict, dev, amp: bool, torch_seed: int):
    """(state with BatchNorm statistics recomputed, the check against `state`).

    pretrain_v2.recompute_bn on the BatchNorm sample, as write_weight_average applies it.
    Called with the same sample and seed for every checkpoint, so the jets, their order,
    the trimmer's random draws and the autocast are the same for all of them."""
    model.load_state_dict(state)
    model.to(dev)
    part = getattr(model, "mod", model)
    own_amp = getattr(part, "use_amp", None)
    past_trimmer_warmup(model)
    PV.seed_all(torch_seed)
    if own_amp is not None:
        # The ParticleTransformer opens its own autocast(enabled=use_amp) inside the
        # outer one, and False turns it off there. Training ran with --use-amp, which
        # sets use_amp; the diagnostic's model is built without it.
        part.use_amp = amp
    try:
        n_bn = PV.recompute_bn(model, sample["batches"], sample["n_jets"], dev, amp, sample["input_names"])
    finally:
        if own_amp is not None:
            part.use_amp = own_amp
    model.eval()
    after = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    check = bn_buffer_check(model, state, after)
    check["batchnorm_layers_recomputed"] = n_bn
    return after, check


def take_jets(batches, n_jets: int, input_names) -> tuple:
    """The first n_jets jets of a stream of (X, y, Z) batches, copied out of the loader's
    shared memory as recompute_bn reads them, with their row ids and native labels."""
    out, rid, lab, seen = [], [], [], 0
    for X, _, Z in batches:
        take = min(len(Z["_rowid"]), n_jets - seen)
        out.append(({k: X[k][:take].clone() for k in input_names}, {}, {"_rowid": Z["_rowid"][:take].clone()}))
        rid.append(np.asarray(Z["_rowid"][:take]).astype("<i8"))
        lab.append(np.asarray(Z["_jet_label"][:take]))      # a tensor, or a masked array the loader left alone
        seen += take
        if seen >= n_jets:
            break
    if seen < n_jets:
        raise SystemExit(f"FATAL: the BatchNorm stream ended after {seen} jets, fewer than {n_jets}")
    return out, np.concatenate(rid), np.concatenate(lab)


def draw_bn_sample(config: str, train: list, n_jets: int, seed: int, num_workers: int):
    """(sample, record): the BatchNorm sample in memory, and what it is."""
    import torch
    sys.path.insert(0, str(REPO / "experiments/MTX"))
    import stream_v2 as sv
    side = PV.sidecar(config)          # exit 42 unless it is the config plus reweighting histograms
    files = PV.to_file_dict(train)
    found = sum(len(v) for v in files.values())
    if found != len(train):
        raise SystemExit(f"FATAL: {found} of {len(train)} BatchNorm training files present")
    ds = sv.StreamDataset(files, side, mode="train", batch_size=BN_STREAM["batch_size"], seed=seed,
                          split_num=BN_STREAM["split_num"], fetch_step=BN_STREAM["fetch_step"],
                          data_fraction=BN_STREAM["data_fraction"])
    ds.set_epoch(BN_STREAM["epoch"])
    names = list(ds.config.input_names)
    kw = {"multiprocessing_context": "fork"} if num_workers else {}
    it = iter(torch.utils.data.DataLoader(ds, batch_size=None, num_workers=num_workers, **kw))
    batches, rid, lab = take_jets(it, n_jets, names)
    del it                              # the workers exit before the test jets are read
    by_index = {v: k for k, v in ds.file_index.items()}
    rec = {"config": config, "config_sha256": PV.sha256_file(config), "sidecar": os.path.basename(side),
           "sidecar_sha256": PV.sha256_file(side), "files_per_family": {k: len(v) for k, v in files.items()},
           "seed_data": seed, "stream": {**BN_STREAM, "num_workers": num_workers}, "n_jets": int(rid.size),
           "rows_sha256": sha256_bytes(np.ascontiguousarray(rid).tobytes()),
           "files": sorted(by_index[i] for i in np.unique(rid >> sv.ROW_BITS).tolist()),
           "qcd_share": float((lab >= sv.NATIVE_QCD_FIRST).mean()), "input_names": names}
    return {"batches": batches, "n_jets": int(rid.size), "input_names": names, "config": ds.config}, rec


def config_differences(a_cfg, b_cfg) -> list:
    """Top-level keys, other than labels, weights and observers, in which two data configs
    differ: the inputs are built the same way when there are none."""
    a, b = a_cfg.options, b_cfg.options
    return sorted(k for k in set(a) | set(b) if k not in ("labels", "weights", "observers") and a.get(k) != b.get(k))


# ---------------------------------------------------------------- infer (GPU)
def infer(a) -> int:
    import torch
    t_start = time.time()
    cuda = torch.cuda.is_available()
    if not cuda and not a.allow_cpu:
        # On the CPU the recompute would run without the training autocast, which is not
        # the v2 model's treatment, and the job would take many hours.
        print("FATAL: torch.cuda.is_available() is False; nothing written", flush=True)
        return EXIT_NO_GPU
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if cuda else "cpu")
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    print(f"device {device} ({gpu}), node {os.environ.get('NODE_NAME', '?')}", flush=True)
    # Every cell from one GPU product: a retry that lands on another product after a run
    # has finished would mix two products in the counts, so it stops the job. The job
    # spec pins one product, so this is a backstop.
    for done in out.glob("*/DONE"):
        was = json.loads(done.read_text()).get("gpu")
        if was != gpu:
            print(f"FATAL: {done.parent.name} was scored on {was}, this pod has {gpu}", flush=True)
            raise SystemExit(PV.EXIT_HALT)

    # The BatchNorm sample first, so the loader's workers have exited before the test jets are held.
    amp = device.type == "cuda"         # the v1 runs trained with --use-amp
    torch_seed = PV.epoch_seed(a.bn_seed, "bn-recompute", 0)
    bn, bn_rec = draw_bn_sample(a.bn_config, a.bn_train, a.bn_jets, a.bn_seed, a.bn_workers)
    bn_rec.update(torch_seed=torch_seed, amp=amp, gpu=gpu,
                  mode="pretrain_v2.recompute_bn: train mode, no gradient, cumulative average (momentum None), "
                       "SequenceTrimmer past its warm-up, generators seeded with torch_seed before each checkpoint",
                  allow_tf32={"matmul": torch.backends.cuda.matmul.allow_tf32,
                              "cudnn": torch.backends.cudnn.allow_tf32})
    rec_path = out / "bn_sample.json"
    if rec_path.exists() and json.loads(rec_path.read_text())["rows_sha256"] != bn_rec["rows_sha256"]:
        raise SystemExit("FATAL: the BatchNorm sample is not the one an earlier attempt drew")
    rec_path.write_text(json.dumps(bn_rec, indent=1))
    print(f"BatchNorm sample: {bn['n_jets']:,} jets from {len(bn_rec['files'])} files, QCD share "
          f"{bn_rec['qcd_share']:.4f}, rows {bn_rec['rows_sha256'][:12]}; {time.time() - t_start:.0f} s", flush=True)

    cfg_path = str(REPO / "configs/data/JetClassII_base.yaml")
    cfg, batches, native, files = HD.read_sample(a, cfg_path)
    diff = config_differences(cfg, bn.pop("config"))
    if diff or list(cfg.input_names) != bn["input_names"]:
        raise SystemExit(f"FATAL: the BatchNorm config builds its inputs differently: {diff}")
    sha = sha256_bytes(native.astype(np.int16).tobytes())
    committed = json.loads(a.reference.read_text())["sample"] if a.reference.exists() else None
    if committed is not None and committed["native_label_sha256"] != sha:
        raise SystemExit(f"FATAL: these are not the jets of {a.reference} ({sha[:12]} vs "
                         f"{committed['native_label_sha256'][:12]})")
    task = HD.P.TASKS[HD.TASK]
    bvc = np.nonzero(np.isin(native, task["signal"] + task["background"]))[0]
    is_qcd = np.isin(native, HD.QCD_NATIVE)
    sample = {"n_jets": int(native.size), "stream_jets": a.max_jets, "stride": a.stride,
              "aligned_with": str(a.align_with), "files": files, "native_label_sha256": sha,
              "matches_committed_sample": str(a.reference) if committed is not None else None,
              "n_qcd": int(is_qcd.sum()), "n_bvc": int(bvc.size), "gpu": gpu,
              "node": os.environ.get("NODE_NAME", "?"), "torch": torch.__version__}
    (out / "sample.json").write_text(json.dumps(sample, indent=1))
    HD._save(out / "sample.npz", native=native, bvc_rows=bvc)
    print(f"{native.size:,} test jets held ({is_qcd.sum():,} QCD, {bvc.size:,} {HD.TASK}); "
          f"{time.time() - t_start:.0f} s", flush=True)

    for spec in a.runs:
        run_dir, rung, k, num_reg = spec.split(":")
        run_dir, k, num_reg = pathlib.Path(run_dir), int(k), int(num_reg)
        rd = out / run_dir.name
        if (rd / "DONE").exists():
            print(f"{run_dir.name}: DONE, skipped", flush=True)
            continue
        rd.mkdir(exist_ok=True)
        lut, qcd, res = HD.vocabulary(rung, k)
        truth = lut[native]
        stored = {f"e{e}": run_dir / f"net_epoch-{e}_state.pt" for e in a.epochs}
        best, best_epoch = run_dir / "net_best_epoch_state.pt", None
        if best.exists():                    # as head_epoch_diag.infer finds it
            b = HD.sha256(best)
            best_epoch = next((e for e in range(a.max_epoch + 1)
                               if (run_dir / f"net_epoch-{e}_state.pt").exists()
                               and HD.sha256(run_dir / f"net_epoch-{e}_state.pt") == b), None)
            if f"e{best_epoch}" not in stored:
                stored["best"] = best

        # The derived states, each written to the state directory and scored from the file.
        # A retry redoes a run without DONE from the start, so nothing there is reused.
        t0 = time.time()
        sd = a.state_dir / run_dir.name
        sd.mkdir(parents=True, exist_ok=True)
        bn_model = HD.XF.build_model(cfg, k + num_reg)
        derived, checks = {}, {}

        def keep(tag, state, check=None):
            path = sd / f"{tag}_state.pt"
            PV.torch_save(state, path)
            derived[tag] = path
            if check is not None:
                if not check["ok"]:
                    raise SystemExit(f"FATAL: {run_dir.name} {tag}: the recompute changed more than "
                                     f"BatchNorm buffers: {check}")
                checks[tag] = check
        for tag, path in stored.items():
            raw = torch.load(str(path), map_location="cpu", weights_only=False)
            keep(f"{tag}_bn", *recompute_state(bn_model, raw.get("model_state_dict", raw), bn, device, amp,
                                               torch_seed))
        paths = [stored[f"e{e}"] for e in a.epochs]
        buf, avg = HD.average_states(paths), PV.average_states(paths)
        agree = buf.keys() == avg.keys() and all(torch.equal(buf[x], avg[x]) for x in buf)
        keep("wavg_buf", buf)
        keep("wavg_bn", *recompute_state(bn_model, avg, bn, device, amp, torch_seed))
        del bn_model
        print(f"{run_dir.name}: {len(checks)} BatchNorm recomputes, {time.time() - t0:.0f} s", flush=True)

        ckpts = {**{t: p for t, p in stored.items() if HD.is_epoch(t)},
                 **{f"{t}_bn": derived[f"{t}_bn"] for t in stored if HD.is_epoch(t)},
                 "wavg_buf": derived["wavg_buf"], "wavg_bn": derived["wavg_bn"]}
        if "best" in stored:
            ckpts.update(best=stored["best"], best_bn=derived["best_bn"])
        meta = {"run_dir": str(run_dir), "rung": rung, "num_classes": k, "num_reg": num_reg,
                "best_epoch": best_epoch, "gpu": gpu, "node": os.environ.get("NODE_NAME", "?"),
                "averages_agree": bool(agree), "bn_checks": checks, "checkpoints": {}}
        for tag, ck in ckpts.items():         # head_epoch_diag.infer's scoring, unchanged
            t0 = time.time()
            model = HD.XF.build_model(cfg, k + num_reg)
            prov = HD.XF.load_trunk_or_die(model, ck, k, num_reg)
            model.to(device).eval()
            tap = HD.XF.ClsTap(model)
            HD.XF.self_check(model, tap, cfg, device)
            logits, feats = HD.run_model(model, tap, batches, device, k, bvc)
            tap.close()
            logp = torch.log_softmax(torch.from_numpy(logits).double(), 1).numpy()
            c, pq_, lo = HD.per_jet(logp, truth, qcd, res)
            HD._save(rd / f"{tag}.npz", correct=c, p_qcd=pq_, log_odds=lo, feat_bvc=feats)
            meta["checkpoints"][tag] = {"path": str(ck), "sha256": prov["sha256"],
                                        "seconds": round(time.time() - t0, 1)}
            print(f"{run_dir.name} {tag}: acc {c.mean():.4f}  {time.time() - t0:.0f} s", flush=True)
        (rd / "DONE").write_text(json.dumps(meta, indent=1))
    print(f"infer finished in {time.time() - t_start:.0f} s", flush=True)
    return 0


# ---------------------------------------------------------------- analyse (CPU)
def repair_report(cks: dict, best) -> dict:
    """Every checkpoint's defect flag under every rule of head_epoch_diag.DEFECT_RULES, and
    what recomputing the BatchNorm statistics did to the stored epochs. `best` is the tag
    of the best-validation checkpoint (an epoch's tag when it is one of them), or None."""
    # The reference is built from the stored epochs alone. Built from the recomputed
    # states, or from all checkpoints, it would move with the treatment it judges, and
    # "the recompute repairs the defect" could come out true by construction.
    ep = sorted((t for t in cks if HD.is_epoch(t)), key=lambda t: int(t[1:]))
    ref = {name: HD.reference_value(cks, ep, m, how) for name, (m, how, *_) in HD.DEFECT_RULES.items()}
    flags = {t: {name: HD.is_defective(c["head"][m], ref[name], d, thr)
                 for name, (m, _, d, thr, _) in HD.DEFECT_RULES.items()} for t, c in cks.items()}
    rules = {}
    for name in HD.DEFECT_RULES:
        bad = [t for t in ep if flags[t][name]]
        rules[name] = {"defective_as_stored": bad,
                       "repaired_by_bn": [t for t in bad if not flags[f"{t}_bn"][name]],
                       "not_repaired_by_bn": [t for t in bad if flags[f"{t}_bn"][name]],
                       "sound_made_defective_by_bn": [t for t in ep if not flags[t][name] and flags[f"{t}_bn"][name]],
                       "wavg_buf_defective": flags["wavg_buf"][name], "wavg_bn_defective": flags["wavg_bn"][name],
                       "best_defective": None if best is None else flags[best][name],
                       "best_bn_defective": None if best is None else flags[f"{best}_bn"][name]}
    return {"reference": ref, "flags": flags, "bn_repair": rules}


def counts(runs: dict) -> dict:
    """Per rule, over the runs: defective stored epochs, what the recompute repaired and
    broke, and which weight averages and best-validation checkpoints are defective."""
    out = {}
    for name in HD.DEFECT_RULES:
        rr = {run: r["bn_repair"][name] for run, r in runs.items()}
        out[name] = {
            "runs": len(rr),
            "runs_with_a_defective_stored_epoch": [x for x, v in rr.items() if v["defective_as_stored"]],
            "defective_stored_epochs": sum(len(v["defective_as_stored"]) for v in rr.values()),
            "repaired_by_bn": sum(len(v["repaired_by_bn"]) for v in rr.values()),
            "runs_with_every_defective_epoch_repaired":
                [x for x, v in rr.items() if v["defective_as_stored"] and not v["not_repaired_by_bn"]],
            "sound_epochs_made_defective_by_bn": sum(len(v["sound_made_defective_by_bn"]) for v in rr.values()),
            "runs_with_a_sound_epoch_made_defective": [x for x, v in rr.items() if v["sound_made_defective_by_bn"]],
            "wavg_buf_defective": [x for x, v in rr.items() if v["wavg_buf_defective"]],
            "wavg_bn_defective": [x for x, v in rr.items() if v["wavg_bn_defective"]],
            "best_defective": [x for x, v in rr.items() if v["best_defective"]],
            "best_bn_defective": [x for x, v in rr.items() if v["best_bn_defective"]]}
    return out


def paired_diff(a: np.ndarray, b: np.ndarray, n_boot: int = DIFF_BOOT, seed: int = 0) -> dict:
    """mean(b) - mean(a) over the same jets, with a paired bootstrap error."""
    d = b.astype(np.float64) - a.astype(np.float64)
    rng = np.random.default_rng(seed)
    boot = np.array([d[rng.integers(0, d.size, d.size)].mean() for _ in range(n_boot)])
    return {"diff": float(d.mean()), "boot_se": float(boot.std(ddof=1)),
            "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "n_boot": n_boot, "seed": seed, "n_jets": int(d.size), "jets_that_differ": int((d != 0).sum())}


def reproducibility(cks: dict, old) -> dict | None:
    """The stored checkpoints re-scored here against the committed v1 diagnostic: top-1
    difference and checkpoint sha256. wavg_buf is that diagnostic's 'wavg': its top-1 is
    compared, its sha256 is not. torch.save writes the file's own name into the archive,
    so the same tensors saved as wavg_buf_state.pt cannot hash the same as v1's
    wavg_e70-79_state.pt."""
    if old is None:
        return None
    stored = [t for t in cks if (HD.is_epoch(t) or t == "best") and t in old["checkpoints"]]
    pairs = {t: t for t in stored}
    if "wavg" in old["checkpoints"]:
        pairs["wavg_buf"] = "wavg"
    diff = {t: abs(cks[t]["head"]["top1_accuracy"] - old["checkpoints"][o]["head"]["top1_accuracy"])
            for t, o in pairs.items()}
    return {"abs_top1_diff": diff, "max_abs_top1_diff": max(diff.values()) if diff else None,
            "sha256_differs": [t for t in stored if old["checkpoints"][t].get("sha256") != cks[t]["sha256"]]}


def analyse(a) -> int:
    out = a.out
    sample = json.loads((out / "sample.json").read_text())
    bn_sample = json.loads((out / "bn_sample.json").read_text())
    z = np.load(out / "sample.npz")
    native, bvc = z["native"], z["bvc_rows"]
    is_qcd = np.isin(native, HD.QCD_NATIVE)
    y = np.isin(native[bvc], HD.P.TASKS[HD.TASK]["signal"]).astype(np.int64)
    committed = json.loads(a.reference.read_text()) if a.reference.exists() else None
    inputs = {n: HD.sha256(out / n) for n in ("sample.json", "sample.npz", "bn_sample.json")}
    if committed is not None:
        inputs["reference"] = {"file": str(a.reference), "sha256": HD.sha256(a.reference)}
    res = {"code": {"repo_ref": os.environ.get("REPO_REF"), "script": "experiments/DIAG/head_bn_diag.py"},
           "inputs": inputs, "sample": sample, "bn_sample": bn_sample, "probe_task": {HD.TASK: HD.P.TASKS[HD.TASK]["names"]},
           "definitions": DEFINITIONS, "primary_rule": HD.PRIMARY_RULE,
           "rules": {name: r[4] for name, r in HD.DEFECT_RULES.items()}, "runs": {}}
    for rd in sorted(p for p in out.iterdir() if (p / "DONE").exists()):
        meta = json.loads((rd / "DONE").read_text())
        inputs[f"{rd.name}/DONE"] = HD.sha256(rd / "DONE")
        cks, correct = {}, {}
        for tag in meta["checkpoints"]:
            d = np.load(rd / f"{tag}.npz")
            probe, _ = HD.probe_summary(d["feat_bvc"], y)
            cks[tag] = {"head": HD.head_summary(d["correct"], d["p_qcd"], d["log_odds"], is_qcd),
                        "probe": probe, "sha256": meta["checkpoints"][tag]["sha256"],
                        "arrays_sha256": HD.sha256(rd / f"{tag}.npz")}
            if tag in ("wavg_buf", "wavg_bn"):
                correct[tag] = d["correct"]
        best = "best" if "best" in cks else (f"e{meta['best_epoch']}" if f"e{meta['best_epoch']}" in cks else None)
        ep = sorted((t for t in cks if HD.is_epoch(t)), key=lambda t: int(t[1:]))
        r = {**{x: meta[x] for x in ("rung", "num_classes", "num_reg", "best_epoch", "gpu", "averages_agree",
                                     "bn_checks")},
             "best_is": best, "checkpoints": cks, **repair_report(cks, best),
             "bn_minus_stored_top1": {t: cks[f"{t}_bn"]["head"]["top1_accuracy"] - cks[t]["head"]["top1_accuracy"]
                                      for t in ep},
             "wavg_bn_minus_wavg_buf_top1": paired_diff(correct["wavg_buf"], correct["wavg_bn"]),
             "reproducibility": reproducibility(cks, (committed or {}).get("runs", {}).get(rd.name))}
        res["runs"][rd.name] = r
        print(f"{rd.name}: {len(cks)} checkpoints", flush=True)
    res["counts"] = counts(res["runs"])
    rep = [r["reproducibility"] for r in res["runs"].values() if r["reproducibility"]]
    res["reproducibility"] = {
        "runs_compared": len(rep),
        "max_abs_top1_diff": max((x["max_abs_top1_diff"] for x in rep if x["max_abs_top1_diff"] is not None),
                                 default=None),
        "sha256_differs": {run: r["reproducibility"]["sha256_differs"] for run, r in res["runs"].items()
                           if r["reproducibility"] and r["reproducibility"]["sha256_differs"]},
        "gpus": sorted({r["gpu"] for r in res["runs"].values()})}
    a.json.write_text(json.dumps(res, indent=1) + "\n")
    print(f"wrote {a.json}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("infer")
    i.add_argument("--runs", nargs="+", required=True, metavar="RUN_DIR:RUNG:K:NUM_REG")
    i.add_argument("--epochs", type=int, nargs="+", default=list(range(70, 80)))
    i.add_argument("--max-epoch", type=int, default=79, help="search 0..this for the best checkpoint's epoch")
    i.add_argument("--data-test", nargs="+", required=True)
    i.add_argument("--max-jets", type=int, required=True)
    i.add_argument("--stride", type=int, required=True)
    i.add_argument("--align-with", type=pathlib.Path, required=True)
    i.add_argument("--batch-size", type=int, default=512)
    i.add_argument("--num-workers", type=int, default=1, help="1 keeps the file order")
    i.add_argument("--bn-config", required=True, help="an arm config; its reweighting sidecar beside it")
    i.add_argument("--bn-train", nargs="+", required=True, metavar="FAMILY:PATH",
                   help="the training files the runs trained on, one path per entry")
    i.add_argument("--bn-jets", type=int, default=PV.BN_JETS)
    i.add_argument("--bn-seed", type=int, default=BN_SEED)
    i.add_argument("--bn-workers", type=int, default=BN_WORKERS)
    i.add_argument("--reference", type=pathlib.Path, default=COMMITTED,
                   help="the committed v1 diagnostic; its sample record must match")
    i.add_argument("--state-dir", type=pathlib.Path, required=True,
                   help="where the derived states are written and scored from; not kept")
    i.add_argument("--allow-cpu", action="store_true",
                   help="tests only: run on the CPU, without autocast, when no GPU is usable")
    i.add_argument("--out", type=pathlib.Path, required=True)
    n = sub.add_parser("analyse")
    n.add_argument("--out", type=pathlib.Path, required=True)
    n.add_argument("--json", type=pathlib.Path, required=True)
    n.add_argument("--reference", type=pathlib.Path, default=COMMITTED,
                   help="the committed v1 diagnostic, for the reproducibility check")
    a = ap.parse_args(argv)
    return {"infer": infer, "analyse": analyse}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
