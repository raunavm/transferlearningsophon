#!/usr/bin/env python3
"""Is `net_best_epoch_state.pt` a better model, or a luckier measurement?

Reads the `run_audit.json` written by scripts/audit_run.py (which carries
`val_metric_by_epoch` and `train_loss_by_epoch`) and asks whether the
epoch-to-epoch validation metric can support ANY checkpoint-selection rule.

Background: weaver scores validation on `--samples-per-epoch-val` samples drawn
from a shuffled 335-file split with persistent workers, so each epoch is scored
on a different subset. `valid_metric` is `total_correct / count` (weaver
utils/nn/tools.py), i.e. plain accuracy on whatever that epoch happened to see.

Run:  python3 scripts/checkpoint_selection_noise.py <run_audit.json> [...]
      kubectl exec -n cms-ml <pod> -- python3 - /data/results/mtx/*/run_audit.json \
          < scripts/checkpoint_selection_noise.py
"""
from __future__ import annotations

import json
import statistics as st
import sys

WINDOW_START = 20  # epochs before this are still genuinely improving


def curve(d, key):
    x = d[key]
    return [x[str(i)] for i in range(len(x))]


def analyse(path):
    d = json.load(open(path))
    val, loss = curve(d, "val_metric_by_epoch"), curve(d, "train_loss_by_epoch")
    w = val[WINDOW_START:]
    mu, sd = st.mean(w), st.pstdev(w)
    last, amax = val[-1], max(w)
    # Under the null that the converged window is exchangeable, `max` is
    # permutation-invariant and only the final position varies, so the p-value
    # for "the last epoch is unusually good" is the fraction at least as high.
    p_last = sum(1 for x in w if x >= last) / len(w)
    return dict(
        run=d["run_id"], K=d["num_classes"],
        loss_first=loss[WINDOW_START], loss_last=loss[-1],
        n=len(w), mean=mu, sd=sd, lo=min(w), hi=max(w),
        val_last=last, val_argmax=d["val_argmax"], argmax_epoch=d["val_argmax_epoch"],
        gap=d["val_argmax"] - last,
        sd_in_units_of_gap=(sd / (d["val_argmax"] - last)) if d["val_argmax"] != last else float("inf"),
        last_percentile=100 * (1 - p_last),
        max_minus_mean_in_sd=(amax - mu) / sd if sd else float("nan"),
    )


def main(paths):
    rows = [analyse(p) for p in paths]
    print(f"{'run':<16}{'K':>4}{'n':>4}{'mean':>8}{'sd':>8}{'range':>16}"
          f"{'last':>8}{'argmax':>8}{'ep':>4}{'gap':>8}{'last %ile':>10}")
    for r in rows:
        print(f"{r['run']:<16}{r['K']:>4}{r['n']:>4}{r['mean']:>8.4f}{r['sd']:>8.4f}"
              f"{f'[{r[chr(108)+chr(111)]:.3f},{r[chr(104)+chr(105)]:.3f}]':>16}"
              f"{r['val_last']:>8.4f}{r['val_argmax']:>8.4f}{r['argmax_epoch']:>4}"
              f"{r['gap']:>8.4f}{r['last_percentile']:>9.0f}%")

    print(f"\ntrain loss over the same window is flat: "
          + ", ".join(f"{r['run'].split('-')[-1]} {r['loss_first']:.3f}->{r['loss_last']:.3f}" for r in rows))

    same = [r for r in rows if r["K"] == max(x["K"] for x in rows)] if len({r["K"] for r in rows}) == 1 else None
    groups = {}
    for r in rows:
        groups.setdefault(r["K"], []).append(r)
    for K, g in sorted(groups.items()):
        if len(g) < 3:
            continue
        sd_arg = st.pstdev([r["val_argmax"] for r in g])
        sd_last = st.pstdev([r["val_last"] for r in g])
        print(f"\nacross the {len(g)} seeds at K={K}:")
        print(f"  SD of the 'best' metric : {sd_arg:.4f}   (a maximum is a stable statistic -- "
              f"it converges on the upper tail, so it LOOKS reproducible)")
        print(f"  SD of the final metric  : {sd_last:.4f}   (a single draw -- unbiased but noisy)")
        print(f"  mean within-run epoch SD: {st.mean([r['sd'] for r in g]):.4f}")
        print(f"  the 'best' metric sits {st.mean([r['max_minus_mean_in_sd'] for r in g]):.2f} SD "
              f"above the run's own mean, which is what selecting a maximum over "
              f"{g[0]['n']} draws buys you for free.")
    print("\nNeither rule yields what the paper needs: the maximum is biased upward by "
          "selection, the final epoch is unbiased but carries the full per-epoch SD. "
          "Both are larger than any arm-to-arm difference we expect to report.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
