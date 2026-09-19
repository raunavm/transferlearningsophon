#!/usr/bin/env python3
"""Turn a masked-particle-modelling checkpoint into a fine-tuning initialisation.

WHY A CONVERTER IS NEEDED AT ALL -- MEASURED 2026-09-18 AGAINST WEAVER 0.4.17.
experiments/MTX/mpm.py wraps the trunk as `MPMNet.trunk`, so every key in its
checkpoint is `trunk.mod.*` (233 tensors) or `decoder.*` (63). The fine-tuning
architecture (experiments/MTX/ParT_sophon_arch_mtx.py) names the same tensors
`mod.*`. weaver 0.4.17's --load-model-weights is a bare
`load_state_dict(strict=False)` after the --exclude-model-weights filter
(train.py:569-585, read from the 0.4.17 wheel) -- it has NO prefix option. The
package installed on the laptop (pip: 0.4.16) does accept `file:prefix`; the
image's 0.4.17 does not, so that form must not be relied on. Offered the raw
file it matches NOTHING: 296 unexpected, every model key missing, and the
fine-tune silently starts from random weights.
The docstrings in mpm.py and ParT_sophon_arch_mpm.py that say the checkpoint
"loads into a supervised arm with strict=False" are wrong about this.

WHICH TENSORS ARE KEPT, AND WHY NOT ALL 233.
    kept     mod.embed.*  mod.pair_embed.*  mod.blocks.*      194 tensors
    dropped  mod.cls_token  mod.cls_blocks.*  mod.norm.*       39 tensors
    dropped  decoder.*                                         63 tensors
MPMNet.encode stops at the end of `mod.blocks`: the two class-attention blocks,
the class token and the final LayerNorm are never called during pretraining, get
no gradient, and the optimiser skips them, so what the checkpoint holds for them
is the pretraining run's random initialisation. Loading that would make the
classifier's starting point a function of the PRETRAINING seed while carrying no
pretraining signal. They are dropped, so they start from the fine-tuning run's
own initialisation -- seeded by the fine-tuning seed, exactly like the head --
which is what the masked-particle-modelling papers do when they attach a fresh
classifier to a pretrained encoder.

THE JOB MUST FAIL LOUDLY IF LESS THAN THE WHOLE TRUNK ARRIVES. Two halves:
here, the count of kept tensors must be exactly N_TRUNK with all eight blocks
present; after weaver loads the file,
`smoke_checks.py load-log --fresh-prefix ... --expect-fresh 39` requires that the
ONLY missing keys are the 39 above plus the head, and that nothing is unexpected.

Run:  python3 experiments/FT/mpm_init.py --src <net_epoch-79_state.pt> --out <file.pt>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re

SRC_PREFIX = "trunk."
KEPT = ("mod.embed.", "mod.pair_embed.", "mod.blocks.")
FRESH = ("mod.cls_token", "mod.cls_blocks.", "mod.norm.")
N_TRUNK, N_FRESH, N_BLOCKS = 194, 39, 8


def convert(state: dict) -> dict:
    if not any(k.startswith("decoder.") for k in state):
        raise SystemExit("FATAL: no decoder.* keys -- this is not a masked-particle-"
                         "modelling checkpoint. A supervised checkpoint put through "
                         "here would silently lose its trained class-attention blocks.")
    trunk = {k[len(SRC_PREFIX):]: v for k, v in state.items() if k.startswith(SRC_PREFIX)}
    stray = [k for k in trunk if not k.startswith(KEPT + FRESH)]
    if stray:
        raise SystemExit(f"FATAL: trunk keys that are neither kept nor known-untrained: "
                         f"{stray[:5]} ({len(stray)}). The architecture changed; decide "
                         f"what each one is before fine-tuning from it.")
    out = {k: v for k, v in trunk.items() if k.startswith(KEPT)}
    n_fresh = sum(1 for k in trunk if k.startswith(FRESH))
    blocks = {int(m.group(1)) for k in out if (m := re.match(r"mod\.blocks\.(\d+)\.", k))}
    if len(out) != N_TRUNK or n_fresh != N_FRESH or blocks != set(range(N_BLOCKS)):
        raise SystemExit(f"FATAL: kept {len(out)} trunk tensors (want {N_TRUNK}), "
                         f"{n_fresh} untrained (want {N_FRESH}), particle-attention "
                         f"blocks {sorted(blocks)} (want 0..{N_BLOCKS - 1}). Refusing to "
                         f"write an init that carries less than the whole trunk.")
    return out


def main(argv=None) -> int:
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    src = pathlib.Path(a.src)
    out = convert(torch.load(src, map_location="cpu"))
    torch.save(out, a.out)
    # Read by smoke_checks.py manifest, so every cell records the source run.
    pathlib.Path(a.out + ".json").write_text(json.dumps({
        "source": str(src),
        "source_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
        "kept_tensors": len(out), "kept_prefixes": list(KEPT),
        "fresh_prefixes": list(FRESH), "fresh_tensors": N_FRESH}, indent=2))
    print(f"wrote {a.out}: {len(out)} particle-attention trunk tensors from {src}; "
          f"{N_FRESH} class-attention tensors and the head start fresh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
