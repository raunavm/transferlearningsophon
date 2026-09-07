#!/usr/bin/env python3
"""MPMv2 self-supervised pretraining for the Sophon trunk, on weaver 0.4.17.

WHAT THIS IS FOR
----------------
`DECISIONS_PENDING.md` item 17 funds one self-supervised arm as the DENOMINATOR
of the granularity claim: without a zero-label floor measured on our own trunk
and protocol, we cannot say what fraction of a granularity effect is attributable
to labels at all. The arm must therefore differ from the supervised arms in
exactly ONE thing -- the pretraining objective -- so the trunk here is the same
8 particle-attention + 2 class-attention, 128-d ParT the arms use, built by the
same `experiments/E1/ParT_sophon_arch_10c.py`, and the data stream is the same
arm config (I2/I3 hold because the `weights:` block is untouched and the labels
are simply never read).

THE RECIPE, AND WHERE EACH PIECE COMES FROM
-------------------------------------------
arXiv:2409.12589 ("Is Tokenization Needed for Masked Particle Modelling?"), read
in full 2026-09-07; the evidence table is in
`research/notes/mpmv2-recipe-corrected-2026-09-07.md`, which also records the
three corrections that reading it forced on the recipe first written into item 17.

  * MAE-style masking, NOT BERT-style. The encoder does not see the dropped
    particles at all; they re-enter only at the decoder (paper Sec. 4).
  * A TRANSFORMER decoder at 1/4 the encoder width. Swapping the MLP decoder for
    a transformer is the single largest gain in the paper's whole ablation
    (regression 63.5 -> 79.2).
  * Positional encoding among the DROPPED particles only -- a distinct mask token
    per p_T rank *within the dropped subset*. Full latent PE "trivializes the
    reconstruction task, which hurts the FM performance" (Sec. 4).
  * Two task heads, summed: L1 on the continuous features, cross-entropy on the
    particle type+charge class. The paper trains every backbone on a continuous
    task TOGETHER WITH the ID task; ID alone is not a configuration it evaluates.
  * 40 % mask rate: the paper's final tuned config (Table 1 last row). Its
    ablation default of 30 % is not what we take.

TWO DELIBERATE DEVIATIONS, both recorded so the paper states them rather than a
referee finding them:

  1. NO REGISTERS. MPMv2 adds 8 learnable register tokens to the encoder, worth
     +1.2/+1.6 accuracy. A register changes the trunk, and I5 fixes the
     architecture across arms; with registers the SSL arm would differ from the
     supervised arms in two things at once and stop being a clean denominator.
  2. MASKING IS APPLIED THROUGH THE ATTENTION MASK, not by compacting the tensor.
     Setting a particle's `mask` bit to 0 removes it from every attention
     computation as a key AND zeroes its embedding, so the encoder output at the
     surviving positions is IDENTICAL to what true removal would give. What we
     give up is only MPMv2's 40 % memory saving, not the objective. The gain is
     that the trunk is called through the same `embed`/`blocks` modules the
     supervised arms use, with no gather/scatter that could silently reorder the
     set -- and memory is not the binding constraint here (the supervised arms
     already fit at 14.8 GB, and the decoder is 1/4 width).

WHY A MONKEYPATCH
-----------------
Same reason as `experiments/MTX/hybrid_mass.py`: the image ships released weaver
0.4.17, which has no SSL train mode and no `get_train_fn` hook that could carry
one. `weaver.train._main` imports `train_classification` INSIDE the function, so
a replacement installed before `weaver.train.main()` is what training runs.

The validation metric is the NEGATIVE total validation loss, because weaver
selects the checkpoint with the MAXIMUM metric and there is no accuracy to report
in a label-free objective.
"""
from __future__ import annotations

import os
import time
from typing import List

import torch
import torch.nn as nn
import tqdm

ENV_FLAG = "MPM_INSTALLED"
DEFAULT_MASK_RATE = 0.40
DEFAULT_DEC_LAYERS = 3
DEFAULT_DEC_HEADS = 4
DEFAULT_ID_WEIGHT = 1.0

# The 8 type+charge classes, in the order the head predicts them. MPMv2 uses
# "8 independent classes" of particle type and charge; our 17 pf_features carry
# exactly the flags needed to reproduce that partition.
ID_CLASSES = ("chad+", "chad-", "nhad", "photon", "ele+", "ele-", "mu+", "mu-")
_TYPE_VARS = ("part_isChargedHadron", "part_isNeutralHadron", "part_isPhoton",
              "part_isElectron", "part_isMuon")
_CHARGE_VAR = "part_charge"
_PT_VAR = "part_pt_scale_log"
# type index -> (class if charge>=0, class if charge<0)
_TYPE_TO_CLASS = ((0, 1), (2, 2), (3, 3), (4, 5), (6, 7))


def feature_groups(var_names: List[str]):
    """Split the arm's pf_features into (continuous idx, type idx, charge idx, pt idx)."""
    missing = [v for v in _TYPE_VARS + (_CHARGE_VAR, _PT_VAR) if v not in var_names]
    if missing:
        raise RuntimeError(
            f"mpm: pf_features is missing {missing}. The MPM arm must use the same "
            f"arm config as the supervised arms; got {tuple(var_names)}")
    type_idx = [var_names.index(v) for v in _TYPE_VARS]
    charge_idx = var_names.index(_CHARGE_VAR)
    cont_idx = [i for i, v in enumerate(var_names)
                if v not in _TYPE_VARS and v != _CHARGE_VAR]
    return cont_idx, type_idx, charge_idx, var_names.index(_PT_VAR)


def id_target(x: torch.Tensor, type_idx, charge_idx) -> torch.Tensor:
    """(N, C, P) features -> (N, P) long in [0, 8).

    The five type flags are one-hot in the released data, so argmax recovers the
    type; charge is the raw -1/0/+1 (the arm config applies no transform to it).
    """
    t = x[:, type_idx, :].argmax(dim=1)                      # (N, P)
    neg = x[:, charge_idx, :] < 0                            # (N, P)
    lut = torch.tensor(_TYPE_TO_CLASS, device=x.device)      # (5, 2)
    return lut[t, neg.long()]


def draw_mask(mask: torch.Tensor, rate: float, generator=None) -> torch.Tensor:
    """Choose which REAL particles to drop. mask/(N,1,P) real=1 -> drop (N,P) bool.

    Per jet, drop round(rate * n_real) particles chosen uniformly without
    replacement, and never all of them: the encoder must keep at least one
    particle or its attention is over an empty set.
    """
    real = mask.squeeze(1).bool()                            # (N, P)
    n_real = real.sum(dim=1)                                 # (N,)
    n_drop = torch.clamp((n_real.float() * rate).round().long(),
                         min=0, max=None)
    n_drop = torch.minimum(n_drop, torch.clamp(n_real - 1, min=0))
    score = torch.rand(real.shape, device=real.device, generator=generator)
    score = score.masked_fill(~real, 2.0)                    # padded sort last
    rank = score.argsort(dim=1).argsort(dim=1)               # 0 = smallest score
    return rank < n_drop.unsqueeze(1)


def drop_pt_rank(pt: torch.Tensor, drop: torch.Tensor, max_drop: int) -> torch.Tensor:
    """p_T rank WITHIN THE DROPPED SUBSET ONLY. pt/drop (N,P) -> rank (N,P) long.

    Rank 0 is the hardest dropped particle. This is the decoder's entire
    positional signal: MPMv2 gives PE "between the masked elements, not the full
    jet", because full latent PE trivializes the reconstruction task. Entries at
    kept positions are meaningless and are never read -- the decoder uses this
    only where `drop` is True.
    """
    big = torch.finfo(pt.dtype).min
    order = pt.masked_fill(~drop, big).argsort(dim=1, descending=True)
    return order.argsort(dim=1).clamp(max=max_drop - 1)


class Decoder(nn.Module):
    """MPMv2's transformer decoder: 1/4 encoder width, fewer layers and heads."""

    def __init__(self, embed_dim: int, dec_dim: int, num_layers: int,
                 num_heads: int, max_drop: int, n_cont: int, activation: str):
        from weaver.nn.model.ParticleTransformer import Block, trunc_normal_
        super().__init__()
        self.proj = nn.Linear(embed_dim, dec_dim)
        # One learnable token per p_T rank AMONG THE DROPPED PARTICLES. This is
        # the whole of the decoder's positional information; the encoder has none.
        self.mask_tokens = nn.Parameter(torch.zeros(max_drop, 1, dec_dim))
        trunc_normal_(self.mask_tokens, std=.02)
        cfg = dict(embed_dim=dec_dim, num_heads=num_heads, ffn_ratio=4,
                   dropout=0.1, attn_dropout=0.1, activation_dropout=0.1,
                   add_bias_kv=False, activation=activation,
                   scale_fc=True, scale_attn=True, scale_heads=True, scale_resids=True)
        self.blocks = nn.ModuleList([Block(**cfg) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(dec_dim)
        self.head_cont = nn.Linear(dec_dim, n_cont)
        self.head_id = nn.Linear(dec_dim, len(ID_CLASSES))
        self.max_drop = max_drop

    def forward(self, z, drop, padding_mask, pt):
        """z (P,N,C) encoder output; drop (N,P) bool; padding_mask (N,P) True=pad."""
        d = self.proj(z)                                     # (P, N, D)
        pdrop = drop.t()                                     # (P, N)
        rank = drop_pt_rank(pt, drop, self.max_drop)         # (N, P)
        tok = self.mask_tokens[rank.t().reshape(-1)].reshape(*pdrop.shape, -1)
        d = torch.where(pdrop.unsqueeze(-1), tok, d)
        for blk in self.blocks:
            d = blk(d, x_cls=None, padding_mask=padding_mask, attn_mask=None)
        d = self.norm(d)
        sel = d.permute(1, 0, 2)[drop]                       # (n_dropped_total, D)
        return self.head_cont(sel), self.head_id(sel)


class MPMNet(nn.Module):
    """The supervised arms' trunk, pretrained by masked reconstruction.

    `self.trunk` is the SAME `ParticleTransformerSophonWrapper` the arms build, so
    a checkpoint written here loads into a supervised arm's model with
    `strict=False` and the trunk weights land in the right places. Only `mod.fc`
    (unused here, and None) and the decoder differ.
    """

    def __init__(self, trunk, cont_idx, type_idx, charge_idx, pt_idx,
                 mask_rate, dec_layers, dec_heads, id_weight, activation='gelu'):
        super().__init__()
        self.trunk = trunk
        self.cont_idx = list(cont_idx)
        self.type_idx = list(type_idx)
        self.charge_idx = int(charge_idx)
        self.pt_idx = int(pt_idx)
        self.mask_rate = float(mask_rate)
        self.id_weight = float(id_weight)
        mod = trunk.mod
        embed_dim = mod.norm.normalized_shape[0]
        self.decoder = Decoder(embed_dim, max(embed_dim // 4, 16), dec_layers,
                               dec_heads, max_drop=128, n_cont=len(self.cont_idx),
                               activation=activation)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'trunk.mod.cls_token', 'decoder.mask_tokens'}

    def encode(self, x, v, mask):
        """ParticleTransformer.forward up to the end of `self.blocks`.

        Replicated rather than called because weaver 0.4.17's ParticleTransformer
        exposes no encoder/aggregator split (the `_forward_encoder` API lives only
        on the dev/custom_train_eval branch -- experiments/E1/ParT_sophon_arch_10c.py
        records the same measurement). Lines below are the 0.4.17 forward verbatim,
        stopping before `cls_blocks`; keep them in sync if the image's weaver moves.
        """
        mod = self.trunk.mod
        padding_mask = ~mask.squeeze(1)                       # (N, P)
        x = mod.embed(x).masked_fill(~mask.permute(2, 0, 1), 0)   # (P, N, C)
        attn_mask = None
        if v is not None and mod.pair_embed is not None:
            attn_mask = mod.pair_embed(v, None).view(-1, v.size(-1), v.size(-1))
        for block in mod.blocks:
            x = block(x, x_cls=None, padding_mask=padding_mask, attn_mask=attn_mask)
        return x, padding_mask

    def forward(self, points, features, lorentz_vectors, mask):
        mod = self.trunk.mod
        with torch.no_grad():
            x, v, mask, _ = mod.trimmer(features, lorentz_vectors, mask, None)
            drop = draw_mask(mask, self.mask_rate)            # (N, P) bool
            tgt_cont = x[:, self.cont_idx, :].permute(0, 2, 1)[drop]
            tgt_id = id_target(x, self.type_idx, self.charge_idx)[drop]
            pt = x[:, self.pt_idx, :]
            # The encoder must not see the dropped particles at all (MAE).
            enc_mask = mask & ~drop.unsqueeze(1)
        z, padding_mask = self.encode(x, v, enc_mask)
        pred_cont, pred_id = self.decoder(z, drop, padding_mask, pt)
        return pred_cont, pred_id, tgt_cont, tgt_id

    def loss(self, pred_cont, pred_id, tgt_cont, tgt_id):
        # float32 on purpose: under autocast the heads emit fp16, and an L1 over
        # standardized features accumulated across ~2600 dropped particles per
        # batch loses real precision in half.
        l_cont = torch.nn.functional.l1_loss(pred_cont.float(), tgt_cont.float())
        l_id = torch.nn.functional.cross_entropy(pred_id.float(), tgt_id)
        return l_cont + self.id_weight * l_id, l_cont, l_id


def _run_epoch(model, opt, scheduler, loader, dev, epoch, train: bool,
               grad_scaler=None, tb_helper=None, steps_per_epoch=None):
    from weaver.utils.logger import _logger

    model.train(train)
    net = model.module if isinstance(
        model, (nn.DataParallel, nn.parallel.DistributedDataParallel)) else model
    data_config = loader.dataset.config
    tot = tot_c = tot_i = 0.0
    tot_correct = n_drop = num_batches = 0
    start = time.time()
    with torch.set_grad_enabled(train), tqdm.tqdm(loader) as tq:
        for X, _y, _z in tq:
            inputs = [X[n].to(dev) for n in data_config.input_names]
            if train:
                opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=grad_scaler is not None):
                pc, pi, tc, ti = model(*inputs)
                loss, l_c, l_i = net.loss(pc, pi, tc, ti)
            if train:
                if grad_scaler is None:
                    loss.backward()
                    opt.step()
                else:
                    grad_scaler.scale(loss).backward()
                    grad_scaler.step(opt)
                    grad_scaler.update()
                if scheduler and getattr(scheduler, '_update_per_step', False):
                    scheduler.step()

            num_batches += 1
            m = ti.shape[0]
            n_drop += m
            tot_correct += (pi.argmax(1) == ti).sum().item()
            tot += loss.item(); tot_c += l_c.item(); tot_i += l_i.item()
            tq.set_postfix({
                'lr': '%.2e' % scheduler.get_last_lr()[0] if scheduler else opt.defaults['lr'],
                'Loss': '%.5f' % loss.item(),
                'L1': '%.5f' % l_c.item(),
                'CE': '%.5f' % l_i.item(),
                'IDAcc': '%.4f' % (tot_correct / max(n_drop, 1)),
                'AvgLoss': '%.5f' % (tot / num_batches)})
            if tb_helper:
                step = tb_helper.batch_train_count + num_batches
                tag = 'train' if train else 'eval'
                tb_helper.write_scalars([
                    (f"LossTot/{tag}", loss.item(), step),
                    (f"LossL1/{tag}", l_c.item(), step),
                    (f"LossCE/{tag}", l_i.item(), step)])
            if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                break

    dt = time.time() - start
    _logger.info('Processed %d dropped particles in %d batches (%.1f s)' % (n_drop, num_batches, dt))
    _logger.info('%s AvgLoss: %.5f, AvgL1: %.5f, AvgCE: %.5f, IDAcc: %.5f (mask_rate=%g)' % (
        'Train' if train else 'Eval', tot / num_batches, tot_c / num_batches,
        tot_i / num_batches, tot_correct / max(n_drop, 1), net.mask_rate))
    if tb_helper:
        tag = 'train' if train else 'eval'
        tb_helper.write_scalars([
            (f"LossTot/{tag} (epoch)", tot / num_batches, epoch),
            (f"IDAcc/{tag} (epoch)", tot_correct / max(n_drop, 1), epoch)])
        if train:
            tb_helper.batch_train_count += num_batches
    if train and scheduler and not getattr(scheduler, '_update_per_step', False):
        scheduler.step()
    return tot / num_batches


def train_mpm(model, loss_func, opt, scheduler, train_loader, dev, epoch,
              steps_per_epoch=None, grad_scaler=None, tb_helper=None):
    _run_epoch(model, opt, scheduler, train_loader, dev, epoch, True,
               grad_scaler, tb_helper, steps_per_epoch)


def evaluate_mpm(model, test_loader, dev, epoch, for_training=True, loss_func=None,
                 steps_per_epoch=None, tb_helper=None, **kwargs):
    if not for_training:
        raise RuntimeError(
            "mpm: this is a self-supervised pretraining objective and has no test "
            "predictions to write. Evaluate the arm by attaching the supervised "
            "head (experiments/MTX/ParT_sophon_arch_mtx.py) to the pretrained trunk "
            "and running the standard eval path.")
    avg = _run_epoch(model, None, None, test_loader, dev, epoch, False,
                     None, tb_helper, steps_per_epoch)
    # weaver keeps the checkpoint with the MAXIMUM validation metric, and a
    # label-free objective has no accuracy; the negative loss preserves the
    # "higher is better" contract without touching weaver's selection rule.
    return -avg


def install(mask_rate: float = DEFAULT_MASK_RATE) -> None:
    """Replace weaver's classification loops with the MPM ones.

    Must run BEFORE weaver.train.main(), and before seed_weaver's lean-val patch.
    """
    import weaver.utils.nn.tools as tools
    tools.train_classification = train_mpm
    tools.evaluate_classification = evaluate_mpm
    os.environ[ENV_FLAG] = "1"
    os.environ["MPM_MASK_RATE"] = repr(float(mask_rate))
