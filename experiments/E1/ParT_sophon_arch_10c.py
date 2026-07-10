"""E1 network config: Sophon architecture (8 particle-attention + 2 class-attention
blocks, pre-activation pair, GELU) for 10-class JetClass-1, used for BOTH arms.

This is a faithful copy of sophon/networks/example_ParticleTransformer_sophon.py
@ commit 9dd6dd6, with the custom `get_train_fn` / `get_evaluate_fn` and their
`train_classification_sophon` / `evaluate_classification_sophon` functions DELETED.
Those hardcode 188-class indices (scores[:, 161:188], Xbb/Xcc, labels['truth_label'])
and crash at 10 classes. With them removed, weaver falls back to its standard
`train_classification` / `evaluate_classification` (weaver/utils/nn/tools.py), which
work with `labels: type: simple`. Architecture (get_model) and loss are unchanged.

Invoke with:  -o num_classes 10 -o fc_params [(512,0.1)]
"""
import torch

from weaver.utils.logger import _logger
from weaver.nn.model.ParticleTransformer import ParticleTransformer


class ParticleTransformerSophonWrapper(torch.nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.export_embed = kwargs.pop('export_embed', False)
        self.mod = ParticleTransformer(**kwargs)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'mod.cls_token', }

    def forward(self, points, features, lorentz_vectors, mask):
        # The image's weaver 0.4.17 ParticleTransformer has NO _forward_encoder /
        # _forward_aggregator split API (that lives in the dev/custom_train_eval branch).
        # The standard forward computes the identical ParT output; E1 needs logits only
        # (no embedding export). This matches the plain ParT wrapper exactly.
        return self.mod(features, v=lorentz_vectors, mask=mask)


def get_model(data_config, **kwargs):
    cfg = dict(
        input_dim=len(data_config.input_dicts['pf_features']),
        num_classes=None,
        # network configurations
        pair_input_dim=4,
        use_pre_activation_pair=True,
        embed_dims=[128, 512, 128],
        pair_embed_dims=[64, 64, 64],
        num_heads=8,
        num_layers=8,
        num_cls_layers=2,
        block_params=None,
        cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
        fc_params=[],
        activation='gelu',
        # misc
        trim=True,
        for_inference=False,
    )
    cfg.update(**kwargs)
    _logger.info('Model config: %s' % str(cfg))

    model = ParticleTransformerSophonWrapper(**cfg)

    model_info = {
        'input_names': list(data_config.input_names),
        'input_shapes': {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names': ['softmax'],
        'dynamic_axes': {**{k: {0: 'N', 2: 'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'softmax': {0: 'N'}}},
    }

    return model, model_info


def get_loss(data_config, **kwargs):
    return torch.nn.CrossEntropyLoss()
