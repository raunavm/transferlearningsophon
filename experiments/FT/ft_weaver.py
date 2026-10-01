#!/usr/bin/env python3
"""experiments/E1/seed_weaver.py, unchanged, with each epoch's validation metric
also written to weaver's log at full precision:

    Epoch #<e>: exact validation metric <repr(float)>

WHY. A fine-tuning cell resumes from its last completed epoch (weaver
--load-epoch; experiments/FT/cell_resume.py), and weaver restarts its
best-validation tracking at 0 in every process, so the best epoch over the whole
run has to be recomputed afterwards. weaver logs the metric with five decimals;
at 199,680 validation jets two accuracies one jet apart round to the same value,
so the five-decimal log cannot always say which epoch weaver itself would have
kept. With the exact value it can: cell_resume.py best applies weaver's own rule
to it.

Nothing else changes: the metric is the value weaver's evaluate returns to its
training loop, the wrapper draws no random numbers, and seed_weaver.py runs as
its own __main__ with the same arguments.

Usage: exactly as experiments/E1/seed_weaver.py.
"""
from __future__ import annotations

import functools
import pathlib
import runpy
import sys

import weaver.utils.nn.tools as tools
from weaver.utils.logger import _logger

SEED_WEAVER = pathlib.Path(__file__).resolve().parents[1] / "E1" / "seed_weaver.py"
_stock = tools.evaluate_classification


# functools.wraps keeps the stock signature visible: seed_weaver.py's
# --lean-val-metrics reads the eval_metrics default from it.
@functools.wraps(_stock)
def _evaluate(model, test_loader, dev, epoch, *args, **kwargs):
    out = _stock(model, test_loader, dev, epoch, *args, **kwargs)
    if not isinstance(out, tuple):        # the training-time validation pass
        _logger.info("Epoch #%d: exact validation metric %r", epoch, float(out))
    return out


def main() -> None:
    tools.evaluate_classification = _evaluate
    sys.argv = [str(SEED_WEAVER)] + sys.argv[1:]
    runpy.run_path(str(SEED_WEAVER), run_name="__main__")


if __name__ == "__main__":
    main()
