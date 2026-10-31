#!/usr/bin/env python3
"""Input variants of the missing-feature control (plan 3.2, 2026-10-31).

The top-tagging data provide four-momenta only, so ten of the seventeen particle features are set
to zero on them (configs/finetune/TopReference.yaml); the quark/gluon data lack the four
track-displacement features (configs/finetune/EnergyFlowQG.yaml). On JetClass-II, where every
feature is present, the same zeroing measures what it costs a frozen representation. Zeroing
removes information and also produces inputs no pretraining jet has: every JetClass-II particle has
exactly one particle-type flag set, so "all flags off" lies off the training manifold. Imputing the
missing features from JetClass-II's own distributions removes the same information but keeps the
inputs on that manifold; comparing the two separates the lost information from the off-manifold
input.

Variants (applied to the standardized pf_features tensor, every slot, before the network):
  mtop  the ten slots the top data lack set to zero (as TopReference.yaml)
  mqg   the four displacement slots set to zero (as EnergyFlowQG.yaml)
  itop  the ten slots imputed: a particle-type flag drawn from JetClass-II's species frequencies
        (exactly one set), a charge drawn given that species (zero for photons and neutral
        hadrons), and the four displacement values of a random JetClass-II particle of the same
        charged/neutral kind
  iqg   the four displacement slots imputed the same way, for the particle's own (kept) charge

The imputation draws depend only on the batch counter and the seed, so every model of a job, and
every job over the same split, receives identical inputs. The reference distributions
(reference(), saved by `python3 experiments/EVAL/input_variants.py ref`) are computed once from
the balanced training set's particles.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np

FLAGS = ("part_isChargedHadron", "part_isNeutralHadron", "part_isPhoton", "part_isElectron", "part_isMuon")
CHARGED_SPECIES = (0, 3, 4)          # indices into FLAGS: charged hadron, electron, muon
DISP = ("part_d0", "part_d0err", "part_dz", "part_dzerr")
TOP_MISSING = ("part_charge",) + FLAGS + DISP
QG_MISSING = DISP
VARIANTS = ("mtop", "mqg", "itop", "iqg")
SEED = 20261031
POOL = 2_000_000


def slots(names: list) -> dict:
    """Index of every relevant feature in the pf_features order of a data config."""
    idx = {n: i for i, n in enumerate(names)}
    missing = [n for n in TOP_MISSING if n not in idx]
    if missing:
        raise SystemExit(f"FATAL: pf_features lacks {missing}")
    return {"charge": idx["part_charge"], "flags": [idx[f] for f in FLAGS], "disp": [idx[d] for d in DISP],
            "top": [idx[n] for n in TOP_MISSING], "qg": [idx[n] for n in QG_MISSING]}


def reference(feats: np.ndarray, mask: np.ndarray, s: dict, rng: np.random.Generator) -> dict:
    """Species frequencies, P(charge > 0 | species) and displacement pools, from standardized
    pf_features (N, F, P) and pf_mask (N, 1, P) of real JetClass-II jets."""
    real = mask[:, 0, :] > 0
    f = np.transpose(feats, (0, 2, 1))[real]                       # (n particles, F)
    flags = f[:, s["flags"]]
    n_set = (flags > 0.5).sum(1)
    if not (n_set == 1).all():
        raise SystemExit(f"FATAL: {int((n_set != 1).sum())} particles without exactly one flag set")
    species = flags.argmax(1)
    counts = np.bincount(species, minlength=len(FLAGS)).astype(np.float64)
    q = f[:, s["charge"]]
    p_pos = np.array([float((q[species == k] > 0).mean()) if (species == k).any() else 0.0
                      for k in range(len(FLAGS))])
    charged = np.isin(species, CHARGED_SPECIES)
    disp = f[:, s["disp"]]
    pool = {}
    for name, sel in (("charged", charged), ("neutral", ~charged)):
        d = disp[sel]
        pool[name] = d[rng.choice(d.shape[0], size=min(POOL, d.shape[0]), replace=False)].astype(np.float32)
    return {"species_p": counts / counts.sum(), "p_charge_pos": p_pos, "pool_charged": pool["charged"],
            "pool_neutral": pool["neutral"], "n_particles": int(f.shape[0])}


class Variant:
    """One input variant: transform(pf_features tensor, batch counter) -> new tensor."""

    def __init__(self, name: str, s: dict, ref: dict | None = None, seed: int = SEED):
        if name not in VARIANTS:
            raise SystemExit(f"FATAL: unknown variant {name}")
        if name.startswith("i") and ref is None:
            raise SystemExit(f"FATAL: variant {name} needs the imputation reference")
        self.name, self.s, self.ref, self.seed = name, s, ref, seed

    def __call__(self, x, counter: int):
        import torch
        x = x.clone()
        if self.name == "mtop":
            x[:, self.s["top"], :] = 0
            return x
        if self.name == "mqg":
            x[:, self.s["qg"], :] = 0
            return x
        g = torch.Generator(device="cpu").manual_seed(self.seed * 1_000_003 + counter)
        n, _, p = x.shape
        ref = self.ref
        if self.name == "itop":
            species = torch.multinomial(torch.as_tensor(ref["species_p"], dtype=torch.float64), n * p,
                                        replacement=True, generator=g).view(n, p)
            for k, i in enumerate(self.s["flags"]):
                x[:, i, :] = (species == k).to(x.dtype).to(x.device)
            pos = torch.as_tensor(ref["p_charge_pos"], dtype=torch.float64)[species]
            sign = torch.where(torch.rand(n, p, generator=g, dtype=torch.float64) < pos, 1.0, -1.0)
            charged = torch.zeros(n, p, dtype=torch.bool)
            for k in CHARGED_SPECIES:
                charged |= species == k
            x[:, self.s["charge"], :] = (sign * charged).to(x.dtype).to(x.device)
        else:
            charged = x[:, self.s["charge"], :].cpu() != 0
        for name, sel in (("pool_charged", charged), ("pool_neutral", ~charged)):
            pool = torch.as_tensor(ref[name])
            draw = pool[torch.randint(0, pool.shape[0], (n, p), generator=g)]      # (n, p, 4)
            d = draw.permute(0, 2, 1).to(x.dtype).to(x.device)                     # (n, 4, p)
            m = sel.to(x.device).unsqueeze(1)
            for j, i in enumerate(self.s["disp"]):
                x[:, i, :] = torch.where(m[:, 0, :], d[:, j, :], x[:, i, :])
        return x


def wrap(model, variant: Variant, feature_index: int):
    """The model with `variant` applied to its pf_features input; `.mod` is the wrapped model's,
    so the feature taps (extract_features.ClsTap, extract_v2.PooledTap) attach to it."""
    import torch

    class VariantModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner, self.counter = model, 0

        @property
        def mod(self):
            return getattr(self.inner, "mod", self.inner)

        def forward(self, *inputs):
            inputs = list(inputs)
            inputs[feature_index] = variant(inputs[feature_index], self.counter)
            self.counter += 1
            return self.inner(*inputs)

    return VariantModel()


def save_ref(path: pathlib.Path, ref: dict, meta: dict) -> None:
    np.savez(path, **{k: np.asarray(v) for k, v in ref.items()})
    path.with_suffix(".json").write_text(json.dumps(
        {**meta, "species": list(FLAGS), "species_p": list(map(float, ref["species_p"])),
         "p_charge_pos": list(map(float, ref["p_charge_pos"])), "n_particles": ref["n_particles"],
         "pool_sizes": [int(ref["pool_charged"].shape[0]), int(ref["pool_neutral"].shape[0])],
         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}, indent=1) + "\n")


def load_ref(path: pathlib.Path) -> dict:
    z = np.load(path)
    return {k: z[k] for k in z.files}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("ref", help="the imputation reference from a JetClass-II file's particles")
    r.add_argument("--data", nargs="+", required=True)
    r.add_argument("--data-config", required=True)
    r.add_argument("--max-jets", type=int, default=100_000)
    r.add_argument("--out", required=True, type=pathlib.Path)
    a = ap.parse_args(argv)
    import torch
    from weaver.utils.data.config import DataConfig
    from weaver.utils.dataset import SimpleIterDataset
    dc = DataConfig.load(a.data_config)
    names = [v[0] for v in dc.input_dicts["pf_features"]]
    s = slots(names)
    ds = SimpleIterDataset({"_": list(a.data)}, a.data_config, for_training=False, fetch_by_files=True,
                           fetch_step=1, name="impute_ref")
    loader = torch.utils.data.DataLoader(ds, batch_size=512, num_workers=1)
    feats, masks, n = [], [], 0
    for X, _y, _z in loader:
        feats.append(X["pf_features"].numpy())
        masks.append(X["pf_mask"].numpy())
        n += feats[-1].shape[0]
        if n >= a.max_jets:
            break
    ref = reference(np.concatenate(feats), np.concatenate(masks), s, np.random.default_rng(SEED))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    save_ref(a.out, ref, {"data": a.data, "data_config": a.data_config, "n_jets": n, "slots": s,
                          "pf_features": names, "seed": SEED})
    print(a.out.with_suffix(".json").read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
