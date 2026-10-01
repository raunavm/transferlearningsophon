"""A numba transcription of build_rand_control.compositions, for the v2 pool scan.

scripts/build_v2_arms.py --select-rand draws thousands of seed-identified random
partitions (build_rand_control.share_draw), and nearly all of a draw's time is
compositions()'s recursive enumeration, 13-30 s per draw in pure Python. This is
the same enumeration -- the same visit order, the same `cap` and the same node
budget, so the same list comes back -- compiled, at about 0.3 s per draw.

It is a local, optional accelerator: numba is in no job image and no cluster
requirement. The builder re-derives every partition that can reach the selection
with the unmodified pure-Python code and refuses on any difference
(build_v2_arms.select_rand), and tests/test_v2_arms.py compares the two on a
fixed sample of seeds.
"""
from __future__ import annotations

import numpy as np
from numba import njit

import build_rand_control as brc


@njit(cache=True)
def _enumerate(avail, vals, target, cap, budget, out, lens):
    """compositions()'s rec(0, target, []) as a loop. Returns how many count
    vectors were written to out[:n] (row r holds lens[r] entries)."""
    n = vals.shape[0]
    cur = np.zeros(n, np.int64)
    cs = np.zeros(n, np.int64)       # the count being tried at each level
    rems = np.zeros(n, np.int64)     # the remainder on entering each level
    nout = 0
    i = 0
    rem = target
    while True:
        # enter rec(i, rem)
        budget -= 1
        if budget < 0:               # every later call returns at once
            return nout
        if rem == 0:
            for j in range(i):
                out[nout, j] = cur[j]
            lens[nout] = i
            nout += 1
        elif i < n and nout < cap:
            v = vals[i]
            hi = rem // v
            if avail[i] < hi:
                hi = avail[i]
            cs[i] = hi
            rems[i] = rem
            cur[i] = hi
            rem -= hi * v
            i += 1
            continue
        # rec(i, .) returned: resume the caller's loop at level i - 1
        while True:
            if i == 0:
                return nout
            i -= 1
            if nout >= cap or cs[i] == 0:
                continue             # that caller returns as well
            cs[i] -= 1
            cur[i] = cs[i]
            rem = rems[i] - cs[i] * vals[i]
            i += 1
            break


def compositions(avail, values, target, cap=brc.COMP_CAP):
    """Drop-in for build_rand_control.compositions: the same list of tuples."""
    vals = np.asarray(values, np.int64)
    av = np.asarray([avail[v] for v in values], np.int64)
    out = np.zeros((cap + 1, len(values)), np.int64)
    lens = np.zeros(cap + 1, np.int64)
    k = _enumerate(av, vals, int(target), int(cap), int(brc.NODE_BUDGET), out, lens)
    return [tuple(int(x) for x in out[r, :lens[r]]) for r in range(k)]


def install() -> None:
    """Swap the compiled enumeration into build_rand_control, in this process."""
    brc.compositions = compositions
