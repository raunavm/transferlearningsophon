"""Wrapper for the official supervised Particle Transformer (ParT).

Loads the public ``ParT_full.pt`` checkpoint from jet-universe/particle_transformer
and exposes both the 10-class logits and the 128-d CLS latent used for
visualization.

ParT architecture matches data/JetClass/JetClass_full.yaml in the official repo.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from weaver.nn.model.ParticleTransformer import ParticleTransformer


PART_KWARGS = dict(
    input_dim=17,
    num_classes=10,
    pair_input_dim=4,
    use_pre_activation_pair=False,
    embed_dims=[128, 512, 128],
    pair_embed_dims=[64, 64, 64],
    num_heads=8,
    num_layers=8,
    num_cls_layers=2,
    activation="gelu",
    fc_params=[],
    trim=True,
    for_inference=False,
)

EMBED_DIM = 128


def _strip_mod_prefix(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    sample = next(iter(state))
    if sample.startswith("mod."):
        return {k[len("mod."):]: v for k, v in state.items()}
    return state


class ParTModel(nn.Module):
    """Official ParT wrapped to return (logits, x_cls)."""

    def __init__(self, checkpoint_path: str | None = None) -> None:
        super().__init__()
        self.mod = ParticleTransformer(**PART_KWARGS)
        if checkpoint_path is not None:
            raw = torch.load(str(checkpoint_path), map_location="cpu",
                             weights_only=False)
            state = raw["model_state_dict"] if "model_state_dict" in raw else raw
            state = _strip_mod_prefix(state)
            missing, unexpected = self.mod.load_state_dict(state, strict=False)
            loaded = len(state) - len(unexpected)
            print(f"Loaded ParT weights from {checkpoint_path} "
                  f"({loaded} loaded, {len(missing)} missing, "
                  f"{len(unexpected)} unexpected)")

        self._has_split_forward = hasattr(self.mod, "_forward_encoder")

    def forward(
        self,
        features: torch.Tensor,
        lorentz_vectors: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features:        (B, 17, N)  ParT input features
            lorentz_vectors: (B, 4,  N)  raw px, py, pz, energy (no scaling)
            mask:            (B, 1,  N)  bool

        Returns:
            logits: (B, 10)
            x_cls:  (B, 128)
        """
        mod = self.mod
        if self._has_split_forward:
            x, padding_mask = mod._forward_encoder(
                features, v=lorentz_vectors, mask=mask
            )
            with torch.amp.autocast("cuda", enabled=getattr(mod, "use_amp", False)):
                x_cls = mod._forward_aggregator(x, padding_mask)
        else:
            x_cls = self._forward_compat(features, lorentz_vectors, mask)
        logits = mod.fc(x_cls)
        return logits, x_cls

    def _forward_compat(self, features, lorentz_vectors, mask):
        """Fallback when weaver doesn't expose split encoder/aggregator."""
        mod = self.mod
        with torch.amp.autocast("cuda", enabled=getattr(mod, "use_amp", False)):
            padding_mask = ~mask.squeeze(1) if mask is not None else None
            if lorentz_vectors is not None and mod.pair_embed is not None:
                attn_mask = mod.pair_embed(lorentz_vectors).view(
                    -1, lorentz_vectors.size(-1), lorentz_vectors.size(-1)
                )
            else:
                attn_mask = None
            x = mod.embed(features)
            if mask is not None:
                x = x.masked_fill(~mask.permute(2, 0, 1), 0)
            for block in mod.blocks:
                x = block(x, x_cls=None, padding_mask=padding_mask,
                          attn_mask=attn_mask)
            cls_tokens = mod.cls_token.expand(-1, x.size(1), -1)
            for block in mod.cls_blocks:
                cls_tokens = block(x, x_cls=cls_tokens, padding_mask=padding_mask)
            x_cls = cls_tokens.squeeze(0)
            if hasattr(mod, "norm") and mod.norm is not None:
                x_cls = mod.norm(x_cls)
        return x_cls
