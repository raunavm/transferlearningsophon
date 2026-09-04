#!/usr/bin/env python3
"""Extract the frozen 128-d representation from a trained arm checkpoint.

WHY THIS IS NOT PART OF THE EVAL JOB
------------------------------------
scripts/build_eval_jobs.py runs `weaver --predict` and caches the K-way SCORES
(pred.root). Those are the head's output. The frozen-probe endpoints in
docs/DOWNSTREAM_SUITE.md are defined on the REPRESENTATION -- the 128-d class
token the head reads -- because a probe fitted on logits measures the head that
the arm's own vocabulary trained, which is the thing being varied. Probing the
representation is what makes the arms comparable at all.

HOW THE REPRESENTATION IS REACHED, AND WHY THIS WAY
---------------------------------------------------
It is NOT reachable through the training path. weaver 0.4.17's
ParticleTransformer has no `_forward_encoder` / `_forward_aggregator` split --
0 occurrences in its source, verified against the v0.4.17 tag -- and
experiments/E1/ParT_sophon_arch_10c.py's wrapper accepts an `export_embed`
kwarg that it pops and never uses, returning logits only. (weaver 0.4.16 DOES
have the split API. Checking the installed 0.4.16 and concluding anything about
the image's 0.4.17 is a mistake this project has now made twice; the tag is the
authority.)

So the representation is taken with a FORWARD PRE-HOOK on `mod.fc`. In 0.4.17's
forward the last two statements are

    x_cls = self.norm(cls_tokens).squeeze(0)
    output = self.fc(x_cls)

so `fc`'s input IS x_cls, by construction. The hook re-implements nothing: the
model runs its own unmodified forward and we read the tensor crossing the
trunk/head boundary. The alternative -- re-implementing the forward pass to
recompute x_cls, as src/models/part_wrapper.py's `_forward_compat` does -- can
silently drift from the real one and produce features that look fine and are
not the model's. `fc` exists in both weaver versions, so the hook is also
version-proof.

TWO CHECKS THAT RUN BEFORE ANY DATA IS READ
-------------------------------------------
Both failure modes here are SILENT -- they yield a plausible AUC from a
meaningless representation -- so both are asserted up front, on random tensors,
in seconds, rather than discovered after hours of streaming:

  1. TRUNK COMPLETENESS. `load_state_dict(strict=False)` is required because the
     head shape depends on K, but it will also happily leave the entire trunk at
     its random initialisation. Every non-`fc` parameter must be accounted for.

  2. HOOK FIDELITY. `fc(captured) == model(...)` to numerical tolerance. If the
     hook grabbed the wrong tensor this fails immediately.

ROW ALIGNMENT ACROSS ARMS
-------------------------
The paired bootstrap in docs/STATISTICS.md compares arms jet-for-jet, so the
feature matrices must be row-aligned. This script therefore always reads
configs/data/JetClassII_base.yaml -- NOT the arm config. The arm configs differ
from the base in the `labels:` block and nothing else, so the base reproduces
their selection, preprocessing and inputs exactly while giving the NATIVE label
(`truth_label: jet_label`, 0..187) that the downstream tasks are defined on. One
shared config, one file order, no shuffling => alignment by construction, and
the saved labels let it be verified rather than assumed.

Usage:
    python3 experiments/EVAL/extract_features.py \
        --checkpoint /data/results/mtx/mtx-l162-s1/net_best_epoch_state.pt \
        --num-classes 162 --arm L162 \
        --data-test '/jc2/jet_data/Res2P_{0250..0299}.parquet' ... \
        --out /data/results/eval/mtx-l162-s1/features
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys
import time

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[2]
EMBED_DIM = 128
# The DEFAULT set, for the granularity arms. The mass-regression work needs
# more than this (genjet_sdmass is its regression TRUTH, not a nice-to-have),
# so the list is a flag -- see --observers. Anything requested must actually
# arrive; see the hard-fail after the loop.
OBSERVERS = ["jet_pt", "jet_sdmass", "jet_eta", "jet_nparticles"]


def load_arch():
    """The ONE model definition, imported rather than copied."""
    import importlib.util
    p = REPO / "experiments" / "MTX" / "ParT_sophon_arch_mtx.py"
    spec = importlib.util.spec_from_file_location("_arch", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_model(data_config, num_classes: int):
    arch = load_arch()
    model, _ = arch.get_model(data_config, num_classes=num_classes,
                              fc_params=[(512, 0.1)])
    return model


def load_trunk_or_die(model, ckpt_path: pathlib.Path, declared_k: int) -> dict:
    """Load a checkpoint and REFUSE to continue on an incomplete or mislabelled one.

    Order matters. The K check runs BEFORE load_state_dict, because
    load_state_dict raises a bare RuntimeError on the head's size mismatch and
    that traceback says nothing about which arm the caller meant. Checking first
    turns it into a sentence.

    strict=False is unavoidable afterwards -- weaver stores the head under the
    same prefix and we want to tolerate nothing else -- and it is also the exact
    mechanism by which a wrong checkpoint yields a randomly-initialised trunk
    with no error at all.
    """
    raw = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state = raw.get("model_state_dict", raw) if isinstance(raw, dict) else raw

    def is_head(k: str) -> bool:
        return ".fc." in f".{k}" or k.startswith("fc.") or k.startswith("mod.fc.")

    # 1. The head's output width IS the checkpoint's K. The trunk is
    #    K-independent, so a K=17 checkpoint declared as K=43 would give
    #    perfectly valid features attributed to the wrong arm -- and the
    #    manifest is what ties a feature matrix to an arm.
    #    The output layer is the HIGHEST-INDEXED Linear in the fc Sequential, not
    #    the widest one: fc_params=[(512,0.1)] gives fc.0.weight of shape
    #    (512, 128) and fc.1.weight of shape (K, 512), so selecting by width
    #    picks the hidden layer and reports K=512 for every arm.
    head_w = []
    for k, v in state.items():
        if not (is_head(k) and k.endswith(".weight") and getattr(v, "ndim", 0) == 2):
            continue
        m = re.search(r"\.(\d+)\.weight$", k)
        head_w.append((int(m.group(1)) if m else -1, k, v))
    ckpt_k = int(max(head_w)[2].shape[0]) if head_w else None
    if ckpt_k is not None and ckpt_k != declared_k:
        print(f"FATAL: {ckpt_path} has a head of width {ckpt_k}, but "
              f"--num-classes says {declared_k}. The trunk is K-independent, so "
              f"this would load cleanly and produce valid features attributed to "
              f"the wrong arm; refusing.", file=sys.stderr)
        raise SystemExit(3)

    # 2. Trunk completeness.
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad_missing = [k for k in missing if not is_head(k)]
    bad_unexpected = [k for k in unexpected if not is_head(k)]
    if bad_missing or bad_unexpected:
        print(f"FATAL: trunk did not load cleanly from {ckpt_path}", file=sys.stderr)
        for k in bad_missing[:10]:
            print(f"  MISSING    {k}", file=sys.stderr)
        for k in bad_unexpected[:10]:
            print(f"  UNEXPECTED {k}", file=sys.stderr)
        print(f"  ({len(bad_missing)} missing, {len(bad_unexpected)} unexpected "
              f"outside the head). A partially-loaded trunk produces a plausible "
              f"AUC from a random representation; refusing.", file=sys.stderr)
        raise SystemExit(2)

    return {
        "checkpoint": str(ckpt_path),
        "checkpoint_num_classes": ckpt_k,
        "sha256": hashlib.sha256(ckpt_path.read_bytes()).hexdigest(),
        "trunk_tensors_loaded": sum(1 for k in state if not is_head(k)),
        "head_missing": len([k for k in missing if is_head(k)]),
        "head_unexpected": len([k for k in unexpected if is_head(k)]),
    }


class ClsTap:
    """Captures `fc`'s input -- which IS x_cls -- via a forward pre-hook."""

    def __init__(self, model):
        fc = getattr(model, "mod", model).fc
        if fc is None:
            raise SystemExit("FATAL: model has fc=None; nothing to tap")
        self.buf: torch.Tensor | None = None
        self.handle = fc.register_forward_pre_hook(self._hook)
        self.fc = fc

    def _hook(self, _module, inputs):
        self.buf = inputs[0].detach()
        return None

    def close(self):
        self.handle.remove()


def synthetic_batch(data_config, batch: int, device):
    """A VALID synthetic batch.

    `torch.randn` for every input makes the forward return NaN, which looks
    exactly like a broken hook and is not one. Two inputs are not free-form:

      *_mask     is a 0/1 indicator. Random values there make the trimmer
                 produce degenerate sequences and the attention softmax reduce
                 over nothing.
      *_vectors  is (px, py, pz, E). The pair embedding forms masses and takes
                 their square root, and a random 4-vector is generally
                 spacelike, so m^2 < 0 and sqrt gives NaN.

    Building E from the momentum makes every particle timelike, which is what a
    real constituent is.
    """
    built = {}
    for name in data_config.input_names:
        s = list(data_config.input_shapes[name])
        s[0] = batch
        if name.endswith("_mask"):
            built[name] = torch.ones(*s, device=device)
        elif name.endswith("_vectors"):
            p = torch.randn(batch, 3, s[-1], device=device)
            e = p.pow(2).sum(1, keepdim=True).sqrt() + 1.0
            built[name] = torch.cat([p, e], dim=1)
        else:
            built[name] = torch.randn(*s, device=device)
    return [built[n] for n in data_config.input_names]


def self_check(model, tap: ClsTap, data_config, device) -> None:
    """Prove the hook captures the representation, on synthetic tensors.

    Runs before any data is read: both failure modes are silent, so they must be
    caught in seconds rather than after hours of streaming.
    """
    model.eval()
    args = synthetic_batch(data_config, 4, device)
    with torch.no_grad():
        out = model(*args)
    if not torch.isfinite(out).all():
        raise SystemExit("FATAL: synthetic forward produced non-finite output; "
                         "the fixture is wrong, not necessarily the hook")
    if tap.buf is None:
        raise SystemExit("FATAL: forward pre-hook on fc never fired")
    if tap.buf.ndim != 2 or tap.buf.shape[1] != EMBED_DIM:
        raise SystemExit(f"FATAL: tapped tensor has shape {tuple(tap.buf.shape)}, "
                         f"expected (N, {EMBED_DIM})")
    with torch.no_grad():
        replay = tap.fc(tap.buf.to(device))
    if not torch.allclose(replay.float(), out.float(), atol=1e-4, rtol=1e-3):
        d = (replay.float() - out.float()).abs().max().item()
        raise SystemExit(f"FATAL: fc(tapped) != model output (max abs diff {d:.3e}). "
                         f"The hook captured the wrong tensor.")
    print(f"  [PASS] hook fidelity: fc(tapped) == model output, "
          f"tapped shape (N, {tap.buf.shape[1]})")
    tap.buf = None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--num-classes", type=int, required=True)
    ap.add_argument("--arm", required=True)
    # Every name here must ALSO be in the data config's `observers:` block --
    # weaver only materialises what that block lists, and the collection loop
    # below skips anything absent from the batch. Silently. That is fine for a
    # decorative kinematic, and not fine for genjet_sdmass, which is the
    # mass-regression target, so a missing observer is fatal rather than absent.
    ap.add_argument("--observers", nargs="+", default=list(OBSERVERS),
                    help="observer branches to cache alongside the features")
    ap.add_argument("--data-test", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-config",
                    default=str(REPO / "configs/data/JetClassII_base.yaml"),
                    help="ALWAYS the shared base config -- see module docstring "
                         "on row alignment. Overridable only for testing.")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--max-jets", type=int, default=0, help="0 = all")
    ap.add_argument("--num-workers", type=int, default=1,
                    help="1, not 2. Each worker holds its own file buffer, so "
                         "workers multiply the dominant memory cost, and a "
                         "single worker also makes the emission order trivially "
                         "deterministic -- which is what row alignment across "
                         "arms rests on.")
    ap.add_argument("--fetch-step", type=int, default=1,
                    help="FILES loaded per fetch (fetch_by_files stays True). "
                         "Training uses 5 for throughput; extraction has no "
                         "reweighting to balance and is memory-bound, so 1.")
    ap.add_argument("--self-check-only", action="store_true",
                    help="build the model, run both guards, write nothing")
    args = ap.parse_args()

    from weaver.utils.dataset import SimpleIterDataset
    from weaver.utils.data.config import DataConfig

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    data_config = DataConfig.load(args.data_config, load_observers=True)
    model = build_model(data_config, args.num_classes)

    print("guards:")
    prov = load_trunk_or_die(model, pathlib.Path(args.checkpoint), args.num_classes)
    print(f"  [PASS] trunk complete: {prov['trunk_tensors_loaded']} tensors loaded, "
          f"0 missing/unexpected outside the head")
    model.to(device).eval()
    tap = ClsTap(model)
    self_check(model, tap, data_config, device)

    if args.self_check_only:
        print("\n--self-check-only: guards passed, nothing written")
        tap.close()
        return 0

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # MEMORY. The model is irrelevant here -- 2.2M parameters is ~9 MB and the
    # accumulated features are 128 floats per jet. What costs memory is the
    # LOADER: fetch_by_files with fetch_step=5 holds five whole JetClass-II
    # files in RAM per worker, which is why training's measured anon band is
    # 35-58 GB. Sizing this job at 32Gi on the theory that "extraction is
    # lighter than training" was wrong in exactly that way, and produced 13
    # OOMKilled pods. fetch_step=1 with one worker cuts the dominant term by
    # roughly ten.
    #
    # fetch_by_files STAYS True. CLAUDE.md is explicit that weaver 0.4.17's
    # event-fraction loader retains ~all loaded jets and climbs without bound;
    # lowering fetch_step within file mode is a different thing entirely and is
    # safe.
    #
    # for_training=False is load-bearing twice over: it disables the flat
    # (pT, m_SD) resampling used in training -- the probe must see the natural
    # test distribution -- AND it is what makes weaver load the observer
    # variables at all.
    #
    # The kwargs are CHECKED against the installed signature rather than
    # assumed. An earlier version passed `load_range_range`, which does not
    # exist in any weaver; the job cloned, pip-installed, loaded the checkpoint,
    # passed both guards and only then died on a TypeError. weaver's dataset API
    # has already shifted under this project once (0.4.16 -> 0.4.17 removed the
    # train/eval hooks and the encoder/aggregator split), so the signature is
    # verified, not trusted. Omitting load_range_and_fraction loads every event,
    # which is what extraction wants.
    import inspect  # noqa: E402
    _params = inspect.signature(SimpleIterDataset.__init__).parameters
    _kw = dict(for_training=False, fetch_by_files=True,
               fetch_step=args.fetch_step, name="extract")
    _unknown = [k for k in _kw if k not in _params]
    if _unknown:
        raise SystemExit(
            f"FATAL: this weaver's SimpleIterDataset has no {_unknown}. "
            f"Accepted: {sorted(k for k in _params if k != 'self')}")
    files = {"_": list(args.data_test)}
    ds = SimpleIterDataset(files, args.data_config, **_kw)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, drop_last=False,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=False)

    observers = list(args.observers)
    feats, labels, obs = [], [], {k: [] for k in observers}
    n, t0 = 0, time.time()
    with torch.no_grad():
        for X, y, Z in loader:
            inputs = [X[k].to(device, non_blocking=True)
                      for k in data_config.input_names]
            model(*inputs)
            feats.append(tap.buf.float().cpu().numpy().astype(np.float32))
            labels.append(y["truth_label"].cpu().numpy().astype(np.int16))
            for k in observers:
                if k in Z:
                    obs[k].append(np.asarray(Z[k]).astype(np.float32))
            n += feats[-1].shape[0]
            if n % (args.batch_size * 200) < args.batch_size:
                print(f"  {n:,} jets  {n/(time.time()-t0):.0f} jets/s", flush=True)
            if args.max_jets and n >= args.max_jets:
                break
    tap.close()

    F = np.concatenate(feats)[: args.max_jets or None]
    L = np.concatenate(labels)[: args.max_jets or None]
    np.save(out / "features.npy", F)
    np.save(out / "label188.npy", L)
    saved_obs = {}
    for k, v in obs.items():
        if v:
            saved_obs[k] = np.concatenate(v)[: args.max_jets or None]
    # A requested observer that never arrived means the data config does not
    # list it. Downstream that reads as "the branch is all zeros" or as a
    # KeyError hours later; here it is one sentence naming the branch.
    absent = [k for k in observers if k not in saved_obs]
    if absent:
        print(f"FATAL: requested observer(s) {absent} never appeared in any "
              f"batch. Add them to the `observers:` block of "
              f"{args.data_config} -- weaver materialises only what it lists.",
              file=sys.stderr)
        raise SystemExit(4)
    if saved_obs:
        np.savez(out / "observers.npz", **saved_obs)

    manifest = {
        "arm": args.arm, "num_classes": args.num_classes,
        "n_jets": int(F.shape[0]), "embed_dim": int(F.shape[1]),
        "data_config": args.data_config,
        "data_config_sha256": hashlib.sha256(
            pathlib.Path(args.data_config).read_bytes()).hexdigest(),
        "n_test_files": len(args.data_test),
        "label188_sha256": hashlib.sha256(L.tobytes()).hexdigest(),
        "observers": sorted(saved_obs),
        **prov,
    }
    (out / "extract_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {F.shape[0]:,} x {F.shape[1]} features to {out}")
    print(f"label188 sha256 {manifest['label188_sha256'][:16]}  "
          f"(must match across arms -- that IS the row-alignment check)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
