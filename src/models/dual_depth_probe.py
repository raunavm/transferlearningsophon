"""Dual-depth embedding extraction at 128-d and 512-d.

Why this exists
---------------
weaver 0.4.17 in the deployed image exposes no `_forward_encoder` /
`_forward_aggregator` split, so there is no supported way to read the
intermediate representation. Dual-depth probing is therefore impossible with
the released API. Two options: run the weaver development branch, or attach
explicit forward hooks. This module does the latter, because it works against
the pinned image and needs no cluster-image rebuild.

The two depths
--------------
    128-d   the class token after the final norm (`mod.norm` output). This is
            the transfer representation, and the one Sophon's X->bs adaptation
            used.
    512-d   the hidden layer inside the classifier head (`mod.fc.0`), i.e. the
            512-wide projection before the K-way logit layer.

Verified head shape from the released checkpoint:
    mod.fc.0.0.weight  (512, 128)      128 -> 512
    mod.fc.1.weight    (188, 512)      512 -> K
so the 512-d probe point is the output of `mod.fc.0`, not a trunk layer.

Usage
-----
    with DualDepthExtractor(model) as ex:
        logits = model(*inputs)
        z128, z512 = ex.embeddings()
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DualDepthExtractor:
    """Context manager attaching forward hooks at the two probe depths.

    Hooks are removed on exit even if the forward pass raises, so a failed
    batch cannot leak hooks into the next one — a leaked hook silently doubles
    memory and returns stale activations.
    """

    def __init__(self, model: nn.Module, detach: bool = True) -> None:
        self.model = model
        self.detach = detach
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._z128: torch.Tensor | None = None
        self._z512: torch.Tensor | None = None

    # ------------------------------------------------------------- resolution
    @staticmethod
    def _resolve(model: nn.Module) -> tuple[nn.Module, nn.Module]:
        """Locate the 128-d and 512-d probe points, or fail loudly."""
        mod = getattr(model, "mod", model)
        norm = getattr(mod, "norm", None)
        fc = getattr(mod, "fc", None)
        if norm is None or fc is None:
            raise AttributeError(
                "DualDepthExtractor: model exposes no `.norm` / `.fc`; "
                f"got {type(mod).__name__}. This module targets the ParT "
                "wrapper used in this project.")
        if not isinstance(fc, nn.Sequential) or len(fc) < 2:
            raise AttributeError(
                "DualDepthExtractor: expected `fc` to be an nn.Sequential with "
                f"a 512-wide hidden block, got {fc!r}. If the head was replaced "
                "with a plain Linear, there is no 512-d probe point and only "
                "the 128-d depth is meaningful.")
        return norm, fc[0]

    # --------------------------------------------------------------- protocol
    def __enter__(self) -> "DualDepthExtractor":
        norm, fc0 = self._resolve(self.model)

        def mk(slot: str):
            def hook(_module, _inp, out):
                t = out.detach() if self.detach else out
                # class token may carry a length-1 sequence axis
                if t.dim() == 3 and t.shape[1] == 1:
                    t = t.squeeze(1)
                setattr(self, slot, t)
            return hook

        self._handles.append(norm.register_forward_hook(mk("_z128")))
        self._handles.append(fc0.register_forward_hook(mk("_z512")))
        return self

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return None

    # ----------------------------------------------------------------- output
    def embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._z128 is None or self._z512 is None:
            raise RuntimeError(
                "DualDepthExtractor: no activations captured — run the forward "
                "pass inside the `with` block before calling embeddings().")
        return self._z128, self._z512

    @property
    def dims(self) -> tuple[int, int]:
        z128, z512 = self.embeddings()
        return z128.shape[-1], z512.shape[-1]


def weaver_has_split_api(module) -> bool:
    """True if this weaver exposes the encoder/aggregator split.

    When False, hooks are required. Checked explicitly so the failure is a
    reported condition rather than an AttributeError deep in a training run.
    """
    return all(hasattr(module, n) for n in ("_forward_encoder", "_forward_aggregator"))
