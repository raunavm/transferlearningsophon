#!/usr/bin/env python3
"""Classification + mass-regression ("hybrid") training for weaver 0.4.17.

WHY A MONKEYPATCH AND NOT A NETWORK-CONFIG HOOK
-----------------------------------------------
The image ships RELEASED weaver 0.4.17, which has no `--train-mode hybrid` and
no `get_train_fn` / `get_evaluate_fn` hook (experiments/E1/seed_weaver.py
records the measurement). GloParT v3 trains its mass head with
`--train-mode hybrid` on colizz/weaver-core-dev
(weaver/scripts/train_GloParT_v3beta4.sh @ 7ac799e), whose utils/nn/tools.py
supplies `train_hybrid` / `evaluate_hybrid`. That fork is not the image's
weaver. This module ports what those two functions do -- ONE shared fc head
carrying K classification nodes plus regression nodes, a log-cosh regression
loss added to the cross-entropy with a weight gamma -- as drop-in replacements
for `weaver.utils.nn.tools.train_classification` / `evaluate_classification`.

`weaver.train._main` imports those two names INSIDE the function
(`from weaver.utils.nn.tools import train_classification as train`), so a
replacement installed before `weaver.train.main()` is what training runs.
seed_weaver.py already relies on exactly this for --lean-val-metrics.

WHAT IS TRAINED
---------------
    model(*inputs) -> (N, K + 1):   [:, :K] class logits,   [:, K] mass node

    loss = CE(logits, truth_label)
         + lambda * mean over valid jets of  logcosh(pred - mass_target)

    mass_target = log(genjet_sdmass / jet_sdmass)
        D2 as amended by DECISIONS_PENDING item 3: GROOMED denominator,
        generator-level soft-drop numerator -- a multiplicative correction
        learned in log space, the meeting's "scale factor times the jet mass".
    mass_valid  = genjet_sdmass > 0
        unmatched is a hard 0.0f (docs/GROUND_TRUTH.md); the log would be -inf.

    lambda = 5.0 by default, GloParT v3beta4's `gamma` (`-o reg_kw {'gamma':5.}`).
    docs/DECISIONS.md D2 asks for a validation fork; docs/RUN_MATRIX.md says to
    cut it to a single value when compute binds, and it binds (item 9).

Both labels come from the data config (`labels.value`; see
scripts/build_arm_configs.py, the *_MASS arms). K is read from `model.num_cls`,
set by experiments/MTX/ParT_sophon_arch_mass.py; a model without it is refused,
because slicing a plain K-wide head at K would leave no mass node at all.

The validation metric is UNCHANGED: accuracy over the K classification nodes,
the same checkpoint-selection rule as the no-mass arms, so the 2x2 contrast
does not also vary the selection rule. The test pass (for_training=False)
returns K-wide softmax scores; the mass node is NOT written to pred.root --
read it from the checkpoint with the mass arch.

log-cosh is GloParT's LogCoshLoss (log(cosh(x))), written here as
x + softplus(-2x) - log 2: the same function without the overflow at |x| > ~45.

The per-batch "Loss/train" scalar and the "Train AvgLoss" log line report the
CLASSIFICATION term alone, so the no-mass arms' loss curves stay comparable;
the regression term and the total are logged beside them.
"""
from __future__ import annotations

import math
import os
import time
from collections import Counter

import torch
import tqdm

LABEL_CLS = "truth_label"
LABEL_REG = "mass_target"
LABEL_VALID = "mass_valid"
DEFAULT_LAMBDA = 5.0
ENV_FLAG = "HYBRID_MASS_INSTALLED"

# weaver's own evaluate_classification, captured once by install(). The hybrid
# evaluate delegates to it on a K-wide view of the model, so validation and
# test go through weaver's code, not a copy of it.
_STOCK_EVAL = None


def logcosh(x: torch.Tensor) -> torch.Tensor:
    """log(cosh(x)), overflow-free: x + softplus(-2x) - log 2."""
    return x + torch.nn.functional.softplus(-2.0 * x) - math.log(2.0)


def masked_logcosh(pred: torch.Tensor, target: torch.Tensor,
                   valid: torch.Tensor) -> torch.Tensor:
    """Mean log-cosh over the jets where `valid` is True; 0 if there are none."""
    w = valid.to(pred.dtype)
    return (logcosh(pred - target) * w).sum() / w.sum().clamp(min=1.0)


def num_cls(model) -> int:
    m = model.module if isinstance(
        model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)) else model
    k = getattr(m, "num_cls", None)
    if k is None:
        raise RuntimeError(
            "hybrid_mass: the model carries no `num_cls`. Train mass arms with "
            "--network-config experiments/MTX/ParT_sophon_arch_mass.py; a plain "
            "K-wide head sliced at K would leave no regression node.")
    return int(k)


def split_output(out: torch.Tensor, k: int):
    if out.ndim != 2 or out.shape[1] != k + 1:
        raise RuntimeError(
            f"hybrid_mass: expected model output (N, {k + 1}) = K logits + 1 "
            f"mass node, got {tuple(out.shape)}")
    return out[:, :k], out[:, k]


class ClsOnly(torch.nn.Module):
    """The model's first K outputs, for weaver's stock evaluation loop."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.k = num_cls(model)

    def forward(self, *inputs):
        return split_output(self.model(*inputs), self.k)[0]


def hybrid_loss(loss_func, out, y, k, lam, dev):
    logits, pred = split_output(out, k)
    label = y[LABEL_CLS].long().to(dev)
    target = y[LABEL_REG].float().to(dev)
    valid = y[LABEL_VALID].bool().to(dev)
    loss_cls = loss_func(logits, label)
    # float32 on purpose: under autocast the head output is fp16 and the
    # log-cosh of a log-ratio does not need the precision loss.
    loss_reg = masked_logcosh(pred.float(), target, valid)
    return loss_cls + lam * loss_reg, loss_cls, loss_reg, logits, label


def make_train_fn(lam: float):
    """weaver 0.4.17's train_classification with the hybrid loss spliced in."""

    def train_hybrid(model, loss_func, opt, scheduler, train_loader, dev, epoch,
                     steps_per_epoch=None, grad_scaler=None, tb_helper=None):
        from weaver.utils.logger import _logger

        model.train()
        data_config = train_loader.dataset.config
        k = num_cls(model)
        for name in (LABEL_CLS, LABEL_REG, LABEL_VALID):
            if name not in data_config.label_names:
                raise RuntimeError(
                    f"hybrid_mass: data config has no label `{name}`; its labels "
                    f"are {tuple(data_config.label_names)}. Use a *_MASS arm config.")

        label_counter = Counter()
        total_loss = total_cls = total_reg = 0.0
        num_batches = total_correct = count = 0
        start_time = time.time()
        with tqdm.tqdm(train_loader) as tq:
            for X, y, _ in tq:
                inputs = [X[n].to(dev) for n in data_config.input_names]
                opt.zero_grad()
                with torch.cuda.amp.autocast(enabled=grad_scaler is not None):
                    out = model(*inputs)
                    loss, loss_cls, loss_reg, logits, label = hybrid_loss(
                        loss_func, out, y, k, lam, dev)
                if grad_scaler is None:
                    loss.backward()
                    opt.step()
                else:
                    grad_scaler.scale(loss).backward()
                    grad_scaler.step(opt)
                    grad_scaler.update()

                if scheduler and getattr(scheduler, '_update_per_step', False):
                    scheduler.step()

                _, preds = logits.max(1)
                n = label.shape[0]
                num_batches += 1
                count += n
                correct = (preds == label).sum().item()
                total_correct += correct
                loss_v, cls_v, reg_v = loss.item(), loss_cls.item(), loss_reg.item()
                total_loss += loss_v
                total_cls += cls_v
                total_reg += reg_v
                label_counter.update(label.numpy(force=True))

                tq.set_postfix({
                    'lr': '%.2e' % scheduler.get_last_lr()[0] if scheduler else opt.defaults['lr'],
                    'Loss': '%.5f' % cls_v,
                    'LossReg': '%.5f' % reg_v,
                    'LossTot': '%.5f' % loss_v,
                    'AvgLoss': '%.5f' % (total_cls / num_batches),
                    'Acc': '%.5f' % (correct / n),
                    'AvgAcc': '%.5f' % (total_correct / count)})

                if tb_helper:
                    tb_helper.write_scalars([
                        ("Loss/train", cls_v, tb_helper.batch_train_count + num_batches),
                        ("LossReg/train", reg_v, tb_helper.batch_train_count + num_batches),
                        ("LossTot/train", loss_v, tb_helper.batch_train_count + num_batches),
                        ("Acc/train", correct / n, tb_helper.batch_train_count + num_batches),
                    ])
                    if tb_helper.custom_fn:
                        with torch.no_grad():
                            tb_helper.custom_fn(model_output=out, model=model,
                                                epoch=epoch, i_batch=num_batches, mode='train')

                if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                    break

        time_diff = time.time() - start_time
        _logger.info('Processed %d entries in total (avg. speed %.1f entries/s)' %
                     (count, count / max(time_diff, 1e-9)))
        _logger.info('Train AvgLoss: %.5f, AvgLossReg: %.5f, AvgLossTot: %.5f, AvgAcc: %.5f (lambda=%g)' %
                     (total_cls / num_batches, total_reg / num_batches,
                      total_loss / num_batches, total_correct / count, lam))
        _logger.info('Train class distribution: \n    %s', str(sorted(label_counter.items())))

        if tb_helper:
            tb_helper.write_scalars([
                ("Loss/train (epoch)", total_cls / num_batches, epoch),
                ("LossReg/train (epoch)", total_reg / num_batches, epoch),
                ("LossTot/train (epoch)", total_loss / num_batches, epoch),
                ("Acc/train (epoch)", total_correct / count, epoch),
            ])
            if tb_helper.custom_fn:
                with torch.no_grad():
                    tb_helper.custom_fn(model_output=out, model=model, epoch=epoch,
                                        i_batch=-1, mode='train')
            tb_helper.batch_train_count += num_batches

        if scheduler and not getattr(scheduler, '_update_per_step', False):
            scheduler.step()

    train_hybrid.hybrid_mass_lambda = float(lam)
    return train_hybrid


def evaluate_hybrid(model, test_loader, dev, epoch, for_training=True, loss_func=None,
                    steps_per_epoch=None,
                    eval_metrics=['roc_auc_score', 'roc_auc_score_matrix', 'confusion_matrix'],
                    tb_helper=None):
    """weaver's evaluate_classification on the K-wide view of the model.

    Same signature and same `eval_metrics` default as the stock function, so
    seed_weaver's --lean-val-metrics patch can wrap this one unchanged.
    """
    if _STOCK_EVAL is None:
        raise RuntimeError("hybrid_mass.install() has not run")
    return _STOCK_EVAL(ClsOnly(model), test_loader, dev, epoch, for_training=for_training,
                       loss_func=loss_func, steps_per_epoch=steps_per_epoch,
                       eval_metrics=eval_metrics, tb_helper=tb_helper)


def install(lam: float = DEFAULT_LAMBDA) -> None:
    """Replace weaver's classification loops with the hybrid ones.

    Must run BEFORE weaver.train.main() and BEFORE seed_weaver's lean-val patch
    (which wraps whatever `evaluate_classification` is at that moment).
    """
    import weaver.utils.nn.tools as tools
    global _STOCK_EVAL
    if _STOCK_EVAL is None:
        _STOCK_EVAL = tools.evaluate_classification
    tools.train_classification = make_train_fn(float(lam))
    tools.evaluate_classification = evaluate_hybrid
    os.environ[ENV_FLAG] = "1"
