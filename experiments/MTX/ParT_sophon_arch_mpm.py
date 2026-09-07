"""MPM network config: the arms' Sophon trunk wrapped for masked-particle pretraining.

The trunk is built by the SAME `experiments/MTX/ParT_sophon_arch_mtx.py` the
supervised arms use (which itself re-exports E1's definition), so there is exactly
one definition of the model and the SSL arm cannot drift from the arms it is the
denominator for. `num_classes=None` and `fc_params=None` drop the classification
head entirely -- weaver's ParticleTransformer then returns the pooled class token
and never builds `fc` -- because MPM trains no classifier. The 2 class-attention
blocks are still constructed and still receive gradients through nothing during
pretraining; they are left in place so the checkpoint's state_dict is a superset
of a supervised arm's trunk and loads into it with `strict=False`.

Invoke with:   -o mask_rate 0.4 -o dec_layers 3 -o dec_heads 4
Train with:    experiments/E1/seed_weaver.py --mpm ...
               which calls experiments/MTX/mpm.install() before weaver.train.main.
               Without it weaver's stock classification loop would run, read a
               label this objective never uses, and train nothing -- so this file
               REFUSES to build unless the loop is installed. Pass
               `-o allow_without_mpm True` only to inspect or export a checkpoint.
"""
import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_mtx = _load("_mpm_mtx_arch", os.path.join(_HERE, "ParT_sophon_arch_mtx.py"))
_mpm = _load("_mpm_impl", os.path.join(_HERE, "mpm.py"))


def get_model(data_config, **kwargs):
    allow = bool(kwargs.pop("allow_without_mpm", False))
    if allow and any(a in ("--predict", "--data-test") for a in sys.argv):
        raise RuntimeError(
            "ParT_sophon_arch_mpm: allow_without_mpm is for checkpoint inspection "
            "only, and this is a --predict/--data-test run. MPM produces no class "
            "scores. Evaluate the pretrained trunk through the supervised head.")
    if not allow and os.environ.get(_mpm.ENV_FLAG) != "1":
        raise RuntimeError(
            "ParT_sophon_arch_mpm: the MPM loop is not installed "
            "(experiments/MTX/mpm.install, via seed_weaver.py --mpm). Refusing to "
            "build an SSL model that weaver's stock classification loop would "
            "silently train against an unused label. For checkpoint inspection "
            "pass -o allow_without_mpm True.")

    mask_rate = float(kwargs.pop("mask_rate", os.environ.get(
        "MPM_MASK_RATE", _mpm.DEFAULT_MASK_RATE)))
    dec_layers = int(kwargs.pop("dec_layers", _mpm.DEFAULT_DEC_LAYERS))
    dec_heads = int(kwargs.pop("dec_heads", _mpm.DEFAULT_DEC_HEADS))
    id_weight = float(kwargs.pop("id_weight", _mpm.DEFAULT_ID_WEIGHT))
    kwargs.pop("num_classes", None)
    kwargs.pop("fc_params", None)

    trunk, info = _mtx.get_model(data_config, num_classes=None, fc_params=None, **kwargs)
    groups = _mpm.feature_groups(list(data_config.input_dicts['pf_features']))
    model = _mpm.MPMNet(trunk, *groups, mask_rate=mask_rate, dec_layers=dec_layers,
                        dec_heads=dec_heads, id_weight=id_weight)
    info['output_names'] = ['mpm']
    return model, info


def get_loss(data_config, **kwargs):
    """weaver requires this. The real loss is MPMNet.loss -- it needs the model's
    own targets, which a (output, label) signature cannot carry. The MPM loop
    never calls what this returns; returning None would crash weaver's setup."""
    import torch
    return torch.nn.CrossEntropyLoss()
