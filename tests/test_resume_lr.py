"""The learning rate after `--load-epoch N` must equal an uninterrupted run's.

Replays weaver 0.4.17's exact order -- scheduler.step() at the end of each
epoch's train(), optimizer state saved AFTER that step, resume constructs
MultiStepLR(last_epoch=N) AFTER opt.load_state_dict() -- for every N, and
compares the rate the resumed run would use for epoch N+1 with the rate the
fresh run used. The stock class fails for N in 55..78 (flat+decay, 80 epochs);
the wrapper must pass for all N and leave a fresh start untouched.
"""
from __future__ import annotations

import copy
import pathlib
import sys
import warnings

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.resume import ResumeSafeMultiStepLR, _STOCK  # noqa: E402

N_EPOCHS, LR0 = 80, 5e-4
N_DECAY = max(1, int(N_EPOCHS * 0.3))
MILESTONES = list(range(N_EPOCHS - N_DECAY, N_EPOCHS))
GAMMA = 0.01 ** (1.0 / N_DECAY)


def _opt():
    return torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=LR0)


def _fresh():
    """lr used while training each epoch, and the optimizer state weaver saves
    after each epoch (i.e. after the scheduler step)."""
    opt = _opt()
    sch = _STOCK(opt, milestones=MILESTONES, gamma=GAMMA)
    used, saved = {}, {}
    for e in range(N_EPOCHS):
        used[e] = opt.param_groups[0]["lr"]
        sch.step()
        saved[e] = copy.deepcopy(opt.state_dict())
    return used, saved


def _resumed_lr(cls, saved_state, n):
    opt = _opt()
    opt.load_state_dict(saved_state)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cls(opt, milestones=MILESTONES, gamma=GAMMA, last_epoch=n)
    return opt.param_groups[0]["lr"]


@pytest.fixture(scope="module")
def fresh():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _fresh()


def test_stock_class_double_decays_inside_the_decay_window(fresh):
    used, saved = fresh
    wrong = [n for n in range(N_EPOCHS - 1)
             if abs(_resumed_lr(_STOCK, saved[n], n) - used[n + 1]) > 1e-15]
    assert wrong == list(range(MILESTONES[0] - 1, N_EPOCHS - 1)), wrong
    ratio = _resumed_lr(_STOCK, saved[55], 55) / used[56]
    assert ratio == pytest.approx(GAMMA)


def test_wrapper_matches_uninterrupted_run_at_every_resume_epoch(fresh):
    used, saved = fresh
    for n in range(N_EPOCHS - 1):
        got = _resumed_lr(ResumeSafeMultiStepLR, saved[n], n)
        assert got == pytest.approx(used[n + 1], rel=0, abs=1e-15), (n, got, used[n + 1])


def test_wrapper_continues_the_schedule_after_resume(fresh):
    used, saved = fresh
    for n in (30, 55, 60, 78):
        opt = _opt()
        opt.load_state_dict(saved[n])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sch = ResumeSafeMultiStepLR(opt, milestones=MILESTONES, gamma=GAMMA, last_epoch=n)
        assert sch.get_last_lr()[0] == pytest.approx(used[n + 1], abs=1e-15)
        for e in range(n + 1, N_EPOCHS):
            assert opt.param_groups[0]["lr"] == pytest.approx(used[e], abs=1e-15), (n, e)
            sch.step()


def test_wrapper_is_a_no_op_for_a_fresh_start():
    a, b = _opt(), _opt()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sa = _STOCK(a, milestones=MILESTONES, gamma=GAMMA)
        sb = ResumeSafeMultiStepLR(b, milestones=MILESTONES, gamma=GAMMA)
        for _ in range(N_EPOCHS):
            assert a.param_groups[0]["lr"] == b.param_groups[0]["lr"]
            sa.step()
            sb.step()
