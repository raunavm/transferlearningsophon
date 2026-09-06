"""Mass-auxiliary network config: E1's Sophon ParT with ONE extra node on the
shared fc head.

THE HEAD, AND WHY IT IS SHAPED LIKE GLOPART'S
---------------------------------------------
GloParT v3beta4 trains one fc head of `num_nodes` outputs of which the first
`num_cls_nodes` are classes and the rest regression targets
(`-o num_nodes 750 -o num_cls_nodes 374`, train_GloParT_v3beta4.sh). The
regression nodes share the hidden layer with the classifier; nothing else in the
network changes. This file does the same with one regression node: the model is
E1's ParticleTransformerSophonWrapper built with `num_classes = K + 1`, and the
first K outputs are the arm's vocabulary exactly as in the no-mass arm.

So the 2x2 {L162, R16_Q1} x {+/- mass} varies ONE thing between a mass arm and
its twin: one extra output column and the loss term that trains it
(experiments/MTX/hybrid_mass.py). Trunk, embedding, hidden layer, dropout,
optimiser, data stream (I2), rate -- identical. The 128-d latent tapped by
experiments/EVAL/extract_features.py is the same tensor in both.

The trunk already sees momentum scale (DECISIONS_PENDING item 2: d0err/dzerr
are absolute-pT-indexed resolution-table cells), and the mass target is a
dimensionless log-ratio, so no kinematic input is added at the head.

Invoke with:   -o num_classes <K> -o fc_params [(512,0.1)]
               K = the arm's vocabulary size, the SAME number the no-mass twin
               is launched with. The model has K + 1 outputs; `model.num_cls`
               records K for hybrid_mass.py.
Train with:    experiments/E1/seed_weaver.py --mass-lambda 5.0 ...
               which installs the hybrid loop. Without it weaver's stock loop
               would softmax over K + 1 nodes and train the mass node as a
               class that is never true -- silently, with a plausible loss
               curve. This file therefore REFUSES to build unless the hybrid
               loop is installed; pass `-o allow_without_hybrid True` only to
               inspect or export a checkpoint.
"""
import importlib.util
import os

_MTX_ARCH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "ParT_sophon_arch_mtx.py")
_spec = importlib.util.spec_from_file_location("_mass_mtx_arch", _MTX_ARCH)
_mtx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mtx)

N_REG = 1
ENV_FLAG = "HYBRID_MASS_INSTALLED"


def get_model(data_config, **kwargs):
    allow = bool(kwargs.pop("allow_without_hybrid", False))
    if not allow and os.environ.get(ENV_FLAG) != "1":
        raise RuntimeError(
            "ParT_sophon_arch_mass: the hybrid class+mass loop is not installed "
            "(experiments/MTX/hybrid_mass.install, via seed_weaver.py "
            "--mass-lambda). Refusing to build a K+1 head that weaver's stock "
            "loop would train as K+1 classes. For checkpoint inspection pass "
            "-o allow_without_hybrid True.")
    if "num_classes" not in kwargs:
        raise RuntimeError("ParT_sophon_arch_mass: pass -o num_classes K (the arm's K)")
    k = int(kwargs.pop("num_classes"))
    model, info = _mtx.get_model(data_config, num_classes=k + N_REG, **kwargs)
    model.num_cls = k
    model.num_reg = N_REG
    return model, info


# The classification term. hybrid_mass adds the regression term around it.
get_loss = _mtx.get_loss
