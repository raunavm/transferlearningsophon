"""GloParT v3's class-gated mass estimator, reproduced from source.

READ THE STRUCTURE BEFORE CHANGING ANYTHING. Built from
colizz/weaver-core-dev @ 7ac799e, NOT from CMS-DP-2026-104, because the note's
formula (p.20) omits the residual composition and reproducing the note verbatim
gives a DIFFERENT, larger-biased estimator than the deployed one.

TRAIN -- example_ParticleTransformer2024PlusTagger_unified2.py,
LogCoshLoss.forward, split_reg=True branch:

    target_cls = one_hot(target_cls, n_cls).bool()
    loss = logcosh(input - target_reg)
    loss = (loss * target_cls).sum(dim=1)     # HARD gate on the TRUE label

and ComposedHybridLoss.forward composes the split node onto the unified one:

    input_reg_split = input_reg_split + input_reg_unifd[:, as_resid_of]

INFER -- example_GloParT3_exporter_final.py:147, deployed `massCorrX2p`:

    massCorrGeneric + (massCorrHbb*probHbb + ... + 2*massCorrHqq*probHqq + ...)
                    / (probHbb + ... + 2*probHqq + ...).clamp(min=1e-10)

so the estimator is

    theta_2p = theta_generic + sum_i a_i p_i delta_i / sum_i a_i p_i

with theta_generic CLASS-AGNOSTIC (trained on every jet, QCD included, hence
on-support for background) and delta_i class-conditional RESIDUALS. QCD nodes are
absent from S -- the note's stated reason is that including them slightly degraded
signal mass resolution.

The gap between hard train-time gating and soft inference-time gating is what this
module exists to expose: `combine` takes the gate explicitly so E5a can rerun the
same trained residuals under oracle routing.
"""
from __future__ import annotations

import numpy as np

CLAMP_MIN = 1e-10          # exporter line 147


def combine(theta_generic, delta, p, alpha=None, classes=None):
    """theta_generic + sum_i a_i p_i delta_i / sum_i a_i p_i.

    theta_generic : (N,)      class-agnostic correction
    delta         : (N, C)    per-class residuals
    p             : (N, C)    gate weights -- predicted probabilities at
                              inference, or a one-hot of the TRUE label for the
                              E5a oracle-routing control
    alpha         : (C,)      per-class weights; GloParT sets 2 on the qq node
    classes       : indices of S. None means every column. QCD columns are
                    excluded by NOT listing them here, exactly as deployed.
    """
    theta_generic = np.asarray(theta_generic, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    if delta.shape != p.shape:
        raise ValueError(f"delta {delta.shape} and p {p.shape} must match")

    if classes is not None:
        delta, p = delta[:, classes], p[:, classes]
        if alpha is not None:
            alpha = np.asarray(alpha, dtype=np.float64)[classes]
    a = np.ones(delta.shape[1]) if alpha is None else np.asarray(alpha, dtype=np.float64)

    w = a[None, :] * p
    num = (w * delta).sum(axis=1)
    den = np.maximum(w.sum(axis=1), CLAMP_MIN)   # exporter uses .clamp(min=1e-10)
    return theta_generic + num / den


def predicted_mass(m_jet, theta):
    """The regressed mass. theta is a multiplicative correction on the UNGROOMED
    jet mass -- CMS-DP-2026-104 p.43, 'the invariant mass computed from all PF
    candidates in the jet'."""
    return np.asarray(m_jet, dtype=np.float64) * np.asarray(theta, dtype=np.float64)


def split_loss_mask(target_cls, n_cls):
    """The train-time hard gate: one-hot on the TRUE label. Only this class's
    residual receives gradient, which is why every delta_i is trained on jets of
    class i alone and is off-support for anything else."""
    out = np.zeros((len(target_cls), n_cls), dtype=bool)
    out[np.arange(len(target_cls)), np.asarray(target_cls)] = True
    return out


def gate_weight(p, alpha=None, classes=None):
    """sum_{i in S} a_i p_i -- how much of the gate a jet puts on the resonance
    hypotheses. For background this measures how far off their training support
    the residuals are being evaluated, and it is the diagnostic published beside
    the E4 headline."""
    p = np.asarray(p, dtype=np.float64)
    if classes is not None:
        p = p[:, classes]
        if alpha is not None:
            alpha = np.asarray(alpha, dtype=np.float64)[classes]
    a = np.ones(p.shape[1]) if alpha is None else np.asarray(alpha, dtype=np.float64)
    return (a[None, :] * p).sum(axis=1)
