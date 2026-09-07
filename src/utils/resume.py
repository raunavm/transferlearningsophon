"""Resume-safe MultiStepLR for weaver 0.4.17's `--load-epoch`.

THE DEFECT (measured 2026-09-07, weaver-core 0.4.17, torch 2.5.1 / 2.9.1)
----------------------------------------------------------------------------
weaver steps the epoch scheduler at the END of `train_classification`
(utils/nn/tools.py: `scheduler.step()` after the batch loop) and saves the
optimizer state AFTER that (train.py main loop: `torch.save(opt.state_dict(),
..._epoch-%d_optimizer.pt)` follows `train(...)`). So the checkpoint written
after epoch N already carries the learning rate that epoch N+1 must train at.
Confirmed on the PVC: `mtx-l162-s1b/net_epoch-54_optimizer.pt` holds 5.000e-4
and `net_epoch-55_optimizer.pt` holds 4.127e-4 = 5e-4 * 0.01**(1/24), i.e. the
milestone-56 decay is inside the epoch-55 file.

On `--load-epoch N`, weaver loads that optimizer state and THEN constructs
`torch.optim.lr_scheduler.MultiStepLR(opt, milestones, gamma, last_epoch=N)`.
Every torch LRScheduler runs `_initial_step()` -> `step()` in its constructor,
which advances `last_epoch` to N+1 and applies `get_lr()`; MultiStepLR's
`get_lr()` multiplies the CURRENT group lr by gamma when `last_epoch` is a
milestone. With flat+decay over 80 epochs the milestones are 56..79, so a
resume from any checkpoint N in 55..78 multiplies by gamma a SECOND time and
the remainder of the run trains at 0.8254x the recipe's rate. Nothing errors
and nothing is logged. Simulated for every N in 0..78 with the exact weaver
order: wrong for N = 55..78, ratio 0.82540, correct elsewhere.

THE FIX
-------
Snapshot each param group's lr before the constructor runs (that value came
from the loaded optimizer state and is already correct for epoch N+1), let the
constructor do its bookkeeping, then put the snapshot back and make
`get_last_lr()` report it. Later `step()` calls behave exactly as in an
uninterrupted run. A fresh start (`last_epoch == -1`) is untouched.

Installed by experiments/E1/seed_weaver.py before `weaver.train.main()`;
weaver looks the class up on `torch.optim.lr_scheduler` at call time, so the
module attribute is what matters. Verified by tests/test_resume_lr.py, which
replays the weaver order for every resume epoch.
"""
from __future__ import annotations

import logging

import torch.optim.lr_scheduler as _lrs

_STOCK = _lrs.MultiStepLR


class ResumeSafeMultiStepLR(_STOCK):
    """MultiStepLR whose construction with `last_epoch != -1` does not re-apply
    a decay that the loaded optimizer state already contains."""

    def __init__(self, optimizer, milestones, gamma=0.1, last_epoch=-1, **kw):
        loaded = None
        if last_epoch != -1:
            loaded = [float(g["lr"]) for g in optimizer.param_groups]
        super().__init__(optimizer, milestones, gamma=gamma, last_epoch=last_epoch, **kw)
        if loaded is None:
            return
        applied = [float(g["lr"]) for g in optimizer.param_groups]
        for g, lr in zip(optimizer.param_groups, loaded):
            g["lr"] = lr
        self._last_lr = list(loaded)
        msg = ("resume-safe MultiStepLR: last_epoch=%d, lr from optimizer state %s "
               "kept (constructor would have set %s)" % (last_epoch, loaded, applied))
        logging.getLogger("weaver").info(msg)
        print("[seed_weaver] " + msg, flush=True)


def install() -> type:
    """Replace torch.optim.lr_scheduler.MultiStepLR; returns the stock class."""
    _lrs.MultiStepLR = ResumeSafeMultiStepLR
    return _STOCK
