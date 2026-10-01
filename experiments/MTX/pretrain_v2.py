#!/usr/bin/env python3
"""v2 pretraining driver for weaver 0.4.17 (audit 2026-09-29, items 1-3).

v1 ran weaver's own `train.py:_main` through experiments/E1/seed_weaver.py.
The audit found (B1, B4, Strand C):
  * the data order and the dropout masks restart from their epoch-0 state at
    every resume, because nothing was derived from the epoch number and weaver
    restores only model and optimizer (weaver train.py:488-500, 822-826);
  * the Lookahead slow weights (Lookahead.state_dict returns only RAdam's
    state, lookahead.py:59-64, and load_state_dict resets the slow weights to
    the fast ones), the AMP GradScaler (created fresh, train.py:811) and
    SequenceTrimmer's warm-up counter (module state outside state_dict) reset
    at every resume;
  * two of the four recorded seeds were inert, and the class token was drawn
    after the output layer, so its value depended on the vocabulary size
    (weaver ParticleTransformer.py:528-529);
  * validation read a rotating reweighted draw from 335 files, so no two epochs
    validated on the same jets, and the best-validation copy was reset by
    every resume (train.py:806);
  * the loader read 5 whole files of one family at a time (stream_v2.py).

This driver keeps weaver's model, loss, optimizer, schedule and data
preprocessing, and replaces the loop around them:
  * trunk_init seeds every trunk tensor (class token included) through a
    trunk-only build; head_init seeds the rest (output layer, mass node, or
    the self-supervised decoder). Neither depends on the output width.
  * data_sampling seeds stream_v2's per-epoch, per-worker, per-fetch draws.
  * dropout reseeds torch, numpy and python at the start of every epoch from
    (dropout seed, epoch): dropout masks, SequenceTrimmer's random trimming and
    the self-supervised masks. Validation reseeds from a fixed value.
  * each epoch writes <out>/metrics/epoch-EEE.json, <out>/stream/epoch-EEE.json,
    net_epoch-E_state.pt and, LAST, net_epoch-E_resume.pt (model, RAdam,
    Lookahead slow weights and counter, GradScaler, scheduler, trimmer counters,
    best-so-far). A restart resumes from the newest complete epoch.
  * the best epoch on the fixed validation sample is copied to
    net_best_epoch_state.pt; --keep-checkpoints decides what else stays.

Run (in the job spec):
  python3 experiments/MTX/pretrain_v2.py --seed S --out /data/results/mtx_v2/RUN \\
      --data-train Res2P:... Res34P:... QCD:... --data-val <25 files> \\
      --data-config configs/arms/ARM.yaml --network-config experiments/MTX/ParT_sophon_arch_mtx.py \\
      -o num_classes K -o fc_params '[(512,0.1)]' --use-amp --batch-size 512 --start-lr 5e-4 \\
      --num-epochs 80 --samples-per-epoch 10240000 --num-workers 5 --fetch-step 1.0 --data-split-num 200
  plus --mass-lambda 5.0 (mass arch) or --mpm [--mpm-mask-rate R] (self-supervised arch).
"""
from __future__ import annotations

import argparse
import ast
import copy
import glob
import hashlib
import inspect
import json
import math
import os
import pathlib
import random
import shutil
import sys
import threading
import time

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

EXIT_HALT = 42          # the job's pod failure policy fails the Job on this code
_MASK63 = (1 << 63) - 1
OBJECTIVES = ("classification", "mass", "mpm")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True, help="run directory")
    ap.add_argument("--data-train", nargs="+", required=True, help="family:path ...")
    ap.add_argument("--data-val", nargs="+", required=True)
    ap.add_argument("--data-config", required=True)
    ap.add_argument("--extra-selection", default=None,
                    help="ANDed onto the config's selection for training and validation, "
                         "after the reweighting histograms (weaver's --extra-selection)")
    ap.add_argument("--network-config", required=True)
    ap.add_argument("-o", "--network-option", nargs=2, action="append", default=[])
    ap.add_argument("--use-amp", action="store_true")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--start-lr", type=float, default=5e-4)
    ap.add_argument("--num-epochs", type=int, default=80)
    ap.add_argument("--samples-per-epoch", type=int, required=True)
    ap.add_argument("--num-workers", type=int, default=5)
    ap.add_argument("--fetch-step", type=float, default=1.0)
    ap.add_argument("--data-split-num", type=int, default=200)
    ap.add_argument("--data-fraction", type=float, default=1.0,
                    help="1/k: each epoch reads a random 1/k of every file, all rows once per k epochs")
    ap.add_argument("--data-windows", type=int, default=None,
                    help="k, in place of --data-fraction 1/k: the same windows in integer arithmetic "
                         "(stream_v2.window_of); recorded as data_fraction 1/k and data_windows k")
    ap.add_argument("--optimizer", default="ranger", choices=["ranger"])
    ap.add_argument("--lr-scheduler", default="flat+decay", choices=["flat+decay"])
    ap.add_argument("--mass-lambda", type=float, default=None)
    ap.add_argument("--mpm", action="store_true")
    ap.add_argument("--mpm-mask-rate", type=float, default=None)
    ap.add_argument("--keep-checkpoints", default="all", choices=["all", "states", "window"],
                    help="all: every epoch's state and resume files (default, ~46 MB per epoch). "
                         "states: every state file and the newest resume file (~9 MB per epoch). "
                         "window: the best epoch, the last ten epochs and the newest resume file.")
    ap.add_argument("--select-on", default="acc", choices=["acc", "head_top1_acc"],
                    help="best-validation metric (first maximum): acc = top-1 weighted by the training "
                         "reweighting weights, Sophon's rule (draft A8, default); head_top1_acc = "
                         "unweighted top-1 on the fixed sample. Both are recorded every epoch. "
                         "Self-supervised: -val.loss.")
    ap.add_argument("--deterministic", action="store_true",
                    help="torch.use_deterministic_algorithms (smoke and numerics checks)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--log-every", type=int, default=2000)
    return ap


# ---------------------------------------------------------------- seeds
def epoch_seed(sub_seed: int, tag: str, epoch: int) -> int:
    """A 63-bit seed for (stream sub-seed, tag, epoch), stable across processes."""
    d = hashlib.sha256(f"seed-stream|v2|{sub_seed}|{tag}|{epoch}".encode()).digest()
    return int.from_bytes(d[:8], "big") & _MASK63


def seed_all(value: int) -> None:
    import numpy as np
    import torch
    random.seed(value)
    np.random.seed(value % 2 ** 32)
    torch.manual_seed(value)           # every CUDA device too


# ---------------------------------------------------------------- model
def _import(path: str, name: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def part_of(model):
    """The ParticleTransformer inside a classifier wrapper or an MPMNet."""
    import torch
    wrapper = model.trunk if isinstance(getattr(model, "trunk", None), torch.nn.Module) else model
    return wrapper.mod


def build_model(network_config: str, data_config, options: dict, seeds: dict):
    """Build the arch's model with the trunk from trunk_init and the rest from
    head_init. Returns (model, loss_func)."""
    from weaver.nn.model.ParticleTransformer import ParticleTransformer
    module = _import(network_config, "_network_module")
    seed_all(seeds["head_init"])
    model, _ = module.get_model(data_config, **copy.deepcopy(options))
    part = part_of(model)
    keys = set(inspect.signature(ParticleTransformer.__init__).parameters) - {"self", "kwargs"}
    trunk_opts = {k: v for k, v in options.items() if k in keys and k not in ("num_classes", "fc_params")}
    base = _import(str(HERE / "ParT_sophon_arch_mtx.py"), "_trunk_only_arch")
    seed_all(seeds["trunk_init"])
    ref, _ = base.get_model(data_config, num_classes=None, fc_params=None, **trunk_opts)
    if ref.mod.fc is not None:
        raise RuntimeError("trunk-only build carries an output layer")
    missing, unexpected = part.load_state_dict(ref.mod.state_dict(), strict=False)
    if unexpected or any(not k.startswith("fc.") for k in missing):
        raise RuntimeError(f"trunk copy mismatch: missing={missing} unexpected={unexpected}")
    loss_func = module.get_loss(data_config, **copy.deepcopy(options))
    return model, loss_func


def trunk_state(model) -> dict:
    return {k: v for k, v in part_of(model).state_dict().items() if not k.startswith("fc.")}


def trimmer_counters(model) -> dict:
    return {n: m._counter for n, m in model.named_modules() if type(m).__name__ == "SequenceTrimmer"}


def set_trimmer_counters(model, counters: dict) -> None:
    mods = dict(model.named_modules())
    for n, c in counters.items():
        mods[n]._counter = c


# ---------------------------------------------------------------- optimizer
def make_optimizer(model, lr: float, num_epochs: int):
    """weaver 0.4.17 train.py optim() for `--optimizer ranger --lr-scheduler
    flat+decay` with no --optimizer-option: Ranger over model.parameters() and
    MultiStepLR over the last 30% of the epochs down to 1% of the rate."""
    import torch
    from weaver.utils.nn.optimizer.ranger import Ranger
    opt = Ranger(model.parameters(), lr=lr)
    n_decay = max(1, int(num_epochs * 0.3))
    milestones = list(range(num_epochs - n_decay, num_epochs))
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=0.01 ** (1.0 / n_decay))
    return opt, sched


def optimizer_state(opt) -> dict:
    """RAdam's state (what weaver saved) plus the Lookahead state it did not."""
    params = [p for g in opt.optimizer.param_groups for p in g["params"]]
    return {"inner": opt.state_dict(),
            "slow": [opt.state[p]["cached_params"].detach().clone() for p in params],
            "step_counter": opt.step_counter, "alpha": opt.alpha, "k": opt.k}


def load_optimizer_state(opt, st: dict) -> None:
    opt.load_state_dict(st["inner"])          # Lookahead.reset(): slow := fast
    params = [p for g in opt.optimizer.param_groups for p in g["params"]]
    if len(params) != len(st["slow"]):
        raise RuntimeError("slow-weight count does not match the model")
    for p, s in zip(params, st["slow"]):
        opt.state[p]["cached_params"].copy_(s)
    opt.step_counter = st["step_counter"]


# ---------------------------------------------------------------- objectives
class Objective:
    """The training loss of one arm and its validation metrics. The train step
    is weaver 0.4.17's train_classification step (utils/nn/tools.py), the hybrid
    step of hybrid_mass.py, or the MPM step of mpm.py, unchanged."""

    def __init__(self, kind, data_config, loss_func, model, lam=None, native_map=None,
                 select_on="acc"):
        self.select_on = select_on
        import torch
        self.kind, self.cfg, self.loss_func, self.lam = kind, data_config, loss_func, lam
        self.label = data_config.label_names[0] if kind == "classification" else None
        if kind == "mass":
            self.hm = _import(str(HERE / "hybrid_mass.py"), "_hybrid_mass_v2")
            self.k = self.hm.num_cls(model)
            self.label = self.hm.LABEL_CLS
        self.qcd_cls = None
        if kind != "mpm" and native_map is not None:
            self.qcd_cls = torch.tensor(sorted(set(native_map[161:].tolist())), dtype=torch.long)

    def train_loss(self, model, inputs, y, dev):
        """-> (loss, {name: detached scalar tensor}, n_correct or None)"""
        if self.kind == "classification":
            label = y[self.label].long().to(dev)
            out = model(*inputs)
            loss = self.loss_func(out, label)
            return loss, {"loss": loss.detach()}, (out.argmax(1) == label).sum()
        if self.kind == "mass":
            out = model(*inputs)
            loss, lc, lr_, logits, label = self.hm.hybrid_loss(self.loss_func, out, y, self.k, self.lam, dev)
            return loss, {"loss": loss.detach(), "loss_cls": lc.detach(), "loss_reg": lr_.detach()}, \
                (logits.argmax(1) == label).sum()
        net = model
        pc, pi, tc, ti = model(*inputs)
        loss, lc, li = net.loss(pc, pi, tc, ti)
        return loss, {"loss": loss.detach(), "loss_l1": lc.detach(), "loss_ce": li.detach()}, None

    def val_batch(self, model, inputs, y, Z, dev, acc: dict):
        import torch
        import torch.nn.functional as F
        if self.kind == "mpm":
            pc, pi, tc, ti = model(*inputs)
            loss, lc, li = model.loss(pc, pi, tc, ti)
            for k, v in (("loss", loss), ("loss_l1", lc), ("loss_ce", li)):
                acc[k] = acc.get(k, 0.0) + v.double()
            acc["id_correct"] = acc.get("id_correct", 0) + (pi.argmax(1) == ti).sum()
            acc["n_drop"] = acc.get("n_drop", 0) + ti.shape[0]
            acc["batches"] = acc.get("batches", 0) + 1
            acc["n"] = acc.get("n", 0) + inputs[0].shape[0]
            return
        out = model(*inputs)
        if self.kind == "mass":
            logits, pred = self.hm.split_output(out, self.k)
            tgt = y[self.hm.LABEL_REG].float().to(dev)
            valid = y[self.hm.LABEL_VALID].bool().to(dev)
            reg = self.hm.logcosh(pred.float() - tgt)
            acc["reg_sum"] = acc.get("reg_sum", 0.0) + (reg * valid).double().sum()
            acc["reg_n"] = acc.get("reg_n", 0) + valid.sum()
        else:
            logits = out
        logits = logits.float()
        label = y[self.label].long().to(dev)
        w = Z["_weight"].to(dev).double()
        ce = F.cross_entropy(logits, label, reduction="none").double()
        correct = (logits.argmax(1) == label).double()
        pq = torch.softmax(logits, 1)[:, self.qcd_cls.to(dev)].sum(1).double()
        is_qcd = (Z["_jet_label"].to(dev) >= 161)
        for k, v in (("n", torch.tensor(float(len(label)), device=dev, dtype=torch.float64)),
                     ("ce", ce.sum()), ("correct", correct.sum()), ("w", w.sum()),
                     ("w_ce", (w * ce).sum()), ("w_correct", (w * correct).sum()),
                     ("n_qcd", is_qcd.double().sum()), ("pq_qcd", (pq * is_qcd).sum()),
                     ("pq_res", (pq * ~is_qcd).sum())):
            acc[k] = acc.get(k, 0.0) + v

    def val_summary(self, acc: dict) -> dict:
        f = {k: (v.item() if hasattr(v, "item") else v) for k, v in acc.items()}
        if self.kind == "mpm":
            b = f["batches"]
            return {"loss": f["loss"] / b, "loss_l1": f["loss_l1"] / b, "loss_ce": f["loss_ce"] / b,
                    "id_acc": f["id_correct"] / max(f["n_drop"], 1), "n_jets": int(f["n"])}
        n, nq = f["n"], f["n_qcd"]
        out = {"acc": f["w_correct"] / f["w"], "loss": f["w_ce"] / f["w"],
               "head_top1_acc": f["correct"] / n, "loss_unweighted": f["ce"] / n,
               "p_qcd_resonant": f["pq_res"] / (n - nq) if n > nq else None,
               "p_qcd_qcd": f["pq_qcd"] / nq if nq else None,
               "n_jets": int(n), "n_qcd": int(nq), "sum_weights": f["w"]}
        if self.kind == "mass":
            out["loss_reg"] = f["reg_sum"] / max(f["reg_n"], 1)
        return out

    def selection(self, val: dict):
        """(name, value): higher is better."""
        if self.kind == "mpm":
            return "-val.loss", -val["loss"]
        return f"val.{self.select_on}", val[self.select_on]


# ---------------------------------------------------------------- records
class StreamRecord:
    """sha256 of the jets one epoch consumed, in the order it consumed them."""

    def __init__(self, n_native: int = 188):
        import numpy as np
        self.h = hashlib.sha256()
        self.n = 0
        self.native = np.zeros(n_native, dtype=np.int64)
        self.native_last = np.zeros(n_native, dtype=np.int64)   # the epoch's last 20% of batches
        self.files = set()

    def update(self, Z, last: bool = False) -> None:
        import numpy as np
        rid = Z["_rowid"].numpy().astype("<i8", copy=False)
        self.h.update(np.ascontiguousarray(rid).tobytes())
        self.n += len(rid)
        c = np.bincount(Z["_jet_label"].numpy(), minlength=len(self.native))[:len(self.native)]
        self.native += c
        if last:
            self.native_last += c
        self.files.update(np.unique(rid >> 20).tolist())

    def record(self, run: str, epoch: int, seed_data: int, seed_dropout: int, files_sha256: str) -> dict:
        rows = self.h.hexdigest()
        return {"run": run, "epoch": epoch, "seed_data": seed_data, "seed_dropout": seed_dropout,
                "files_sha256": files_sha256, "rows_sha256": rows,
                "sha256": hashlib.sha256((files_sha256 + rows).encode()).hexdigest(),
                "n_jets": int(self.n)}


def _anon_bytes():
    for path, key in (("/sys/fs/cgroup/memory.stat", "anon"),
                      ("/sys/fs/cgroup/memory/memory.stat", "total_rss")):
        try:
            with open(path) as f:
                for line in f:
                    k, v = line.split()
                    if k == key:
                        return int(v)
        except OSError:
            continue
    return None


class MemMonitor(threading.Thread):
    """Peak cgroup anonymous memory (all processes of the pod), sampled every 5 s."""

    def __init__(self, period: float = 5.0):
        super().__init__(daemon=True)
        self.period, self.peak = period, None
        self.start()

    def run(self):
        while True:
            v = _anon_bytes()
            if v is not None:
                self.peak = v if self.peak is None else max(self.peak, v)
            time.sleep(self.period)

    def take(self):
        p, self.peak = self.peak, None
        return None if p is None else round(p / 2 ** 30, 2)


# ---------------------------------------------------------------- weight average (amendment A8)
WAVG_EPOCHS = 10          # the robustness checkpoint: the last ten epochs (70-79 of 80)
BN_JETS = 200_000         # BatchNorm statistics recomputed on this many training jets


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def average_states(paths) -> dict:
    """Mean of each floating tensor over the state dicts; other tensors (the
    BatchNorm batch counters) from the last one."""
    import torch
    states = [torch.load(p, map_location="cpu", weights_only=True) for p in paths]
    out = {}
    for k, v in states[-1].items():
        if v.is_floating_point():
            out[k] = (sum(s[k].double() for s in states) / len(states)).to(v.dtype)
        else:
            out[k] = v.clone()
    return out


def recompute_bn(model, batches, n_jets: int, dev, amp: bool, input_names) -> int:
    """Running statistics of every BatchNorm layer from scratch: reset, cumulative
    average (momentum None), forward passes in train mode without gradients over
    the first n_jets jets of `batches`; nothing else changes. Returns the number
    of BatchNorm layers."""
    import torch
    bns = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    momenta = [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None
    model.train()
    seen = 0
    with torch.no_grad():
        for X, y, Z in batches:
            take = min(len(Z["_rowid"]), n_jets - seen)
            inputs = [X[k][:take].to(dev, non_blocking=True) for k in input_names]
            with torch.cuda.amp.autocast(enabled=amp):
                model(*inputs)
            seen += take
            if seen >= n_jets:
                break
    for m, mom in zip(bns, momenta):
        m.momentum = mom
    return len(bns)


def write_weight_average(out: pathlib.Path, a, model, ds_train, seeds, dev, amp, loader_kw, input_names):
    """net_wavg<F>-<L>_state.pt: the weight average of the last WAVG_EPOCHS epochs'
    state files (70-79 of 80), BatchNorm statistics recomputed on BN_JETS training jets
    drawn by the stream of epoch num_epochs (the epoch-80 stream; its rows are training
    rows, read in earlier epochs too), and
    net_wavg<F>-<L>.json recording the inputs' sha256, the file's sha256 and the
    BatchNorm sample. The format is the one experiments/FT/ft_v2.py resolve_wavg reads."""
    import torch
    from torch.utils.data import DataLoader
    first = max(0, a.num_epochs - WAVG_EPOCHS)
    epochs = list(range(first, a.num_epochs))
    name = f"net_wavg{first}-{a.num_epochs - 1}"
    meta = out / f"{name}.json"
    if meta.exists():
        return
    paths = [out / f"net_epoch-{e}_state.pt" for e in epochs]
    model.load_state_dict(average_states(paths))
    seed_all(epoch_seed(seeds["dropout"], "bn-recompute", 0))
    ds_train.set_epoch(a.num_epochs)
    rec = StreamRecord()

    def recorded(it):
        seen = 0
        for X, y, Z in it:
            take = min(len(Z["_rowid"]), BN_JETS - seen)
            rec.update({k: v[:take] for k, v in Z.items()})
            seen += take
            yield X, y, Z
    it = iter(DataLoader(ds_train, **loader_kw))
    n_bn = recompute_bn(model, recorded(it), BN_JETS, dev, amp, input_names)
    del it
    state = out / f"{name}_state.pt"
    torch_save(model.state_dict(), state)
    names = {v: k for k, v in ds_train.file_index.items()}
    write_json(meta, {
        "inputs": {str(e): sha256_file(p) for e, p in zip(epochs, paths)},
        "sha256": sha256_file(state),
        "bn_recompute": {"n_jets": int(rec.n), "batchnorm_layers": n_bn, "stream_epoch": a.num_epochs,
                         "seed_data": seeds["data_sampling"], "rows_sha256": rec.h.hexdigest(),
                         "files": sorted(names[i] for i in rec.files),
                         "dropout_seed": epoch_seed(seeds["dropout"], "bn-recompute", 0),
                         "mode": "train, no gradient, cumulative average (momentum None)"},
        "code": {"repo_ref": os.environ.get("REPO_REF"), "driver": "experiments/MTX/pretrain_v2.py"}})
    print(f"[pretrain_v2] {state.name}: mean of epochs {first}-{a.num_epochs - 1}, "
          f"{n_bn} BatchNorm layers recomputed on {rec.n} jets", flush=True)


def write_json(path: pathlib.Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1) + "\n")
    os.replace(tmp, path)


def torch_save(obj, path: pathlib.Path) -> None:
    import torch
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def latest_complete_epoch(out: pathlib.Path):
    """Newest epoch whose resume file (written last) and state file exist."""
    best = None
    for p in out.glob("net_epoch-*_resume.pt"):
        e = int(p.name[len("net_epoch-"):-len("_resume.pt")])
        if (out / f"net_epoch-{e}_state.pt").exists() and (best is None or e > best):
            best = e
    return best


def prune(out: pathlib.Path, keep_epochs, newest_resume: int) -> None:
    """Delete resume files other than the newest and, unless keep_epochs is
    None, state files of epochs not in keep_epochs. The newest resume epoch's
    state file always stays: latest_complete_epoch needs both to restart there."""
    for p in out.glob("net_epoch-*_state.pt"):
        e = int(p.name[len("net_epoch-"):-len("_state.pt")])
        if keep_epochs is not None and e not in keep_epochs and e != newest_resume:
            p.unlink()
    for p in out.glob("net_epoch-*_resume.pt"):
        e = int(p.name[len("net_epoch-"):-len("_resume.pt")])
        if e != newest_resume:
            p.unlink()


# ---------------------------------------------------------------- data
def to_file_dict(entries) -> dict:
    """weaver train.py to_filelist(): 'family:glob' or a bare path (family '_')."""
    out = {}
    for f in entries:
        name, fp = f.split(":", 1) if ":" in f else ("_", f)
        out.setdefault(name, []).extend(glob.glob(fp))
    return {k: sorted(v) for k, v in out.items()}


def sidecar(path: str) -> str:
    """The arm config's reweighting sidecar, refused (exit 42) unless it is the
    config plus reweighting histograms. Never rebuilt here."""
    import stream_v2 as sv
    side = sv.sidecar_path(path)
    if not os.path.exists(side):
        print(f"FATAL: no reweighting sidecar {side}", flush=True)
        raise SystemExit(EXIT_HALT)
    bad = sv.sidecar_mismatch(path, side)
    if bad:
        print(f"FATAL: {side} is not {path} plus reweighting histograms: differs in {bad}", flush=True)
        raise SystemExit(EXIT_HALT)
    return side


def load_data_config(path: str):
    import stream_v2 as sv
    return sv.load_config(sidecar(path))


def recipe_of(a) -> dict:
    keep = ("seed", "data_config", "extra_selection", "network_config", "network_option", "use_amp", "batch_size",
            "start_lr", "num_epochs", "samples_per_epoch", "num_workers", "fetch_step",
            "data_split_num", "data_fraction", "data_windows", "optimizer", "lr_scheduler", "mass_lambda", "mpm", "mpm_mask_rate",
            "deterministic", "select_on")
    r = {k: getattr(a, k) for k in keep}
    r["data_train_n"] = {k: len(v) for k, v in to_file_dict(a.data_train).items()}
    r["data_val"] = sorted(os.path.basename(p) for p in to_file_dict(a.data_val).get("_", []))
    r["device"] = device_name(a.device)
    r["code"] = code_version()
    return r


def code_version() -> dict:
    """The tag the job cloned (REPO_REF) and the commit checked out: resuming under
    other code would blend two programs into one run."""
    import subprocess
    try:
        commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True,
                                text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {"repo_ref": os.environ.get("REPO_REF"), "commit": commit}


def stream_window(a) -> dict:
    """The window arguments of the training stream: the integer count when given
    (data_fraction is then only its record), else the fraction."""
    if a.data_windows is not None:
        return {"data_windows": a.data_windows}
    return {"data_fraction": a.data_fraction}


def device_name(device=None) -> str:
    """Resuming on another GPU product would blend two numerics into one run."""
    import torch
    if (device or ("cuda" if torch.cuda.is_available() else "cpu")).startswith("cuda"):
        return torch.cuda.get_device_name(0)
    return "cpu"


# ---------------------------------------------------------------- main
def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    if a.mpm and a.mass_lambda is not None:
        raise SystemExit("pretrain_v2: --mpm and --mass-lambda are exclusive")
    if a.mpm_mask_rate is not None and not a.mpm:
        raise SystemExit("pretrain_v2: --mpm-mask-rate without --mpm")
    if a.data_windows is not None:
        if a.data_fraction != 1.0 or a.data_windows < 1:
            raise SystemExit("pretrain_v2: --data-windows k replaces --data-fraction 1/k")
        a.data_fraction = 1.0 / a.data_windows        # recorded; the stream uses the integer count
    if a.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from src.utils.reproducibility import derive_all
    sys.path.insert(0, str(HERE))
    import stream_v2 as sv

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if a.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)

    kind = "mpm" if a.mpm else ("mass" if a.mass_lambda is not None else "classification")
    if kind == "mass":
        os.environ["HYBRID_MASS_INSTALLED"] = "1"
    if kind == "mpm":
        mpm = _import(str(HERE / "mpm.py"), "_mpm_v2")
        rate = a.mpm_mask_rate if a.mpm_mask_rate is not None else mpm.DEFAULT_MASK_RATE
        if not 0.0 < rate < 1.0:
            raise SystemExit("pretrain_v2: --mpm-mask-rate must be in (0, 1)")
        os.environ[mpm.ENV_FLAG] = "1"
        os.environ["MPM_MASK_RATE"] = repr(float(rate))

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    run = out.name
    recipe = recipe_of(a)
    rpath = out / "recipe.json"
    if rpath.exists() and json.loads(rpath.read_text()) != recipe:
        print(f"FATAL: {out} holds a different recipe:\n  on disk {rpath.read_text()}\n"
              f"  this run {json.dumps(recipe)}", flush=True)
        return EXIT_HALT
    write_json(rpath, recipe)

    seeds = derive_all(a.seed)
    print(f"[pretrain_v2] run={run} seed={a.seed} objective={kind} seeds={seeds}", flush=True)
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    side = sidecar(a.data_config)
    data_config = sv.load_config(side, a.extra_selection)
    train_files = to_file_dict(a.data_train)
    val_files = to_file_dict(a.data_val)
    ds_train = sv.StreamDataset(train_files, side, mode="train", batch_size=a.batch_size,
                                seed=seeds["data_sampling"], split_num=a.data_split_num,
                                fetch_step=a.fetch_step, extra_selection=a.extra_selection,
                                **stream_window(a))
    ds_val = sv.StreamDataset(val_files, side, mode="val", batch_size=a.batch_size,
                              extra_selection=a.extra_selection)

    options = {k: ast.literal_eval(v) for k, v in a.network_option}
    if a.use_amp:
        options["use_amp"] = True
    if kind == "mpm":
        options["mask_rate"] = float(os.environ["MPM_MASK_RATE"])
    model, loss_func = build_model(a.network_config, data_config, options, seeds)
    model = model.to(dev)
    obj = Objective(kind, data_config, loss_func, model, lam=a.mass_lambda, select_on=a.select_on,
                    native_map=sv.native_to_class(data_config) if kind != "mpm" else None)
    opt, sched = make_optimizer(model, a.start_lr, a.num_epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=a.use_amp and dev.type == "cuda")
    amp = a.use_amp and dev.type == "cuda"
    steps = a.samples_per_epoch // a.batch_size
    best = {"epoch": None, "metric": None, "value": None}

    start = 0
    last = latest_complete_epoch(out)
    if last is not None:
        st = torch.load(out / f"net_epoch-{last}_resume.pt", map_location=dev, weights_only=False)
        model.load_state_dict(st["model"])
        load_optimizer_state(opt, st["optimizer"])
        sched.load_state_dict(st["scheduler"])
        if st["scaler"]:
            scaler.load_state_dict(st["scaler"])
        set_trimmer_counters(model, st["trimmer_counters"])
        best = st["best"]
        start = last + 1
        print(f"[pretrain_v2] resumed after epoch {last}: lr={opt.param_groups[0]['lr']:.4e} "
              f"lookahead_counter={opt.step_counter} scale={scaler.get_scale() if amp else None} "
              f"best={best}", flush=True)
    else:
        print("[pretrain_v2] fresh start", flush=True)
        torch_save({"trunk": trunk_state(model)}, out / "init_trunk.pt")

    mem = MemMonitor()
    loader_kw = dict(batch_size=None, num_workers=a.num_workers, pin_memory=dev.type == "cuda",
                     persistent_workers=False)
    if a.num_workers:
        loader_kw["multiprocessing_context"] = "fork"    # Linux's default, explicit for laptops
    input_names = list(data_config.input_names)
    for epoch in range(start, a.num_epochs):
        lr = opt.param_groups[0]["lr"]
        seed_all(epoch_seed(seeds["dropout"], "train", epoch))
        ds_train.set_epoch(epoch)
        rec = StreamRecord()
        model.train()
        sums, n_correct, t0 = {}, 0, time.time()
        it = iter(DataLoader(ds_train, **loader_kw))
        for step in range(steps):
            X, y, Z = next(it)
            if step == 0:
                t_first, n_first, max_fetch = time.time(), len(Z["_rowid"]), 0
            max_fetch = max(max_fetch, int(Z["_fetch"].max()))
            rec.update(Z, last=step >= int(0.8 * steps))
            inputs = [X[k].to(dev, non_blocking=True) for k in input_names]
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=amp):
                loss, parts, corr = obj.train_loss(model, inputs, y, dev)
            if amp:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            for k, v in parts.items():
                sums[k] = sums.get(k, 0.0) + v.double()
            if corr is not None:
                n_correct = n_correct + corr
            if a.log_every and (step + 1) % a.log_every == 0:
                print(f"[pretrain_v2] epoch {epoch} step {step + 1}/{steps} "
                      f"avg loss {sums['loss'].item() / (step + 1):.5f} "
                      f"{rec.n / (time.time() - t0):.0f} jets/s", flush=True)
        del it
        sched.step()
        t_train = time.time() - t0
        train = {k: v.item() / steps for k, v in sums.items()}
        if not all(math.isfinite(v) for v in train.values()):
            print(f"FATAL: non-finite training loss at epoch {epoch}: {train}", flush=True)
            return EXIT_HALT
        if kind != "mpm":
            train["acc"] = float(n_correct) / rec.n
        train.update(n_jets=rec.n, qcd_share=float(rec.native[161:].sum()) / rec.n,
                     native_counts=rec.native.tolist(), native_counts_last20=rec.native_last.tolist(),
                     qcd_share_last20=float(rec.native_last[161:].sum()) / max(int(rec.native_last.sum()), 1),
                     seconds=round(t_train, 1),
                     jets_per_s=round(rec.n / t_train, 1),
                     startup_seconds=round(t_first - t0, 1), max_fetch_id=max_fetch,
                     jets_per_s_after_first_batch=round((rec.n - n_first) / max(t0 + t_train - t_first, 1e-9), 1))

        seed_all(epoch_seed(seeds["dropout"], "validation", 0))
        model.eval()
        t1, vacc = time.time(), {}
        with torch.no_grad():
            for X, y, Z in DataLoader(ds_val, **loader_kw):
                inputs = [X[k].to(dev, non_blocking=True) for k in input_names]
                with torch.cuda.amp.autocast(enabled=amp):
                    obj.val_batch(model, inputs, y, Z, dev, vacc)
        val = obj.val_summary(vacc)
        val["seconds"] = round(time.time() - t1, 1)
        name, value = obj.selection(val)
        is_best = best["value"] is None or value > best["value"]
        if is_best:
            best = {"epoch": epoch, "metric": name, "value": value}

        files_sha = sv.plan_sha256(train_files, seeds["data_sampling"], epoch, a.num_workers,
                                   a.data_split_num, a.fetch_step, a.data_fraction, a.data_windows)
        stream = rec.record(run, epoch, seeds["data_sampling"], seeds["dropout"], files_sha)
        write_json(out / "metrics" / f"epoch-{epoch:03d}.json", {
            "run": run, "epoch": epoch, "objective": kind, "lr": lr, "train": train, "val": val,
            "selection": {"metric": name, "value": value, "is_best": is_best, "best": best},
            "stream_sha256": stream["sha256"], "peak_anon_gb": mem.take(),
            "device": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"})
        write_json(out / "stream" / f"epoch-{epoch:03d}.json", stream)
        torch_save(model.state_dict(), out / f"net_epoch-{epoch}_state.pt")
        if is_best:
            shutil.copy2(out / f"net_epoch-{epoch}_state.pt", out / "net_best_epoch_state.tmp")
            os.replace(out / "net_best_epoch_state.tmp", out / "net_best_epoch_state.pt")
            write_json(out / "best_epoch.json", best)
        torch_save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer_state(opt),
                    "scheduler": sched.state_dict(), "scaler": scaler.state_dict() if amp else {},
                    "trimmer_counters": trimmer_counters(model), "best": best},
                   out / f"net_epoch-{epoch}_resume.pt")
        if a.keep_checkpoints == "window":
            prune(out, {best["epoch"]} | set(range(a.num_epochs - 10, a.num_epochs)), epoch)
        elif a.keep_checkpoints == "states":
            prune(out, None, epoch)
        print(f"[pretrain_v2] epoch {epoch} done: lr {lr:.4e} train {train['loss']:.5f} "
              f"val {name}={value:.5f} best={best['epoch']} stream {stream['sha256'][:12]} "
              f"{train['jets_per_s']:.0f} jets/s", flush=True)
    write_weight_average(out, a, model, ds_train, seeds, dev, amp, loader_kw, input_names)
    (out / "DONE").write_text(json.dumps(best) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
