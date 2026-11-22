#!/usr/bin/env python3
"""Files written during the storage faults of 17-19 Nov 2026 that hold only NUL bytes (2026-11-21: one mass file was 59 KB
of zeros, written 2026-11-18 00:41). Reads the first 4 KB of every non-empty results file modified in the window and lists
those that are all zeros. Reads only; writes the list.

    python3 experiments/EVAL/zero_fill_scan.py --out /data/results/eval/zero_fill_scan.txt
"""
import argparse
import os
import time

ROOTS = ("/data/results/eval", "/data/results/ft_v2", "/data/results/bootstrap")
LO, HI = time.mktime((2026, 11, 17, 0, 0, 0, 0, 0, -1)), time.mktime((2026, 11, 19, 12, 0, 0, 0, 0, -1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    n, bad = 0, []
    for root in ROOTS:
        for dp, _dn, fn in os.walk(root):
            for f in fn:
                p = os.path.join(dp, f)
                try:
                    st = os.stat(p)
                except OSError:
                    continue
                if st.st_size == 0 or not LO <= st.st_mtime <= HI:
                    continue
                n += 1
                with open(p, "rb") as fh:
                    head = fh.read(4096)
                if head.count(0) == len(head):
                    bad.append(f"{p}\t{st.st_size}")
    with open(a.out, "w") as fh:
        fh.write(f"# {n} files modified 2026-11-17 00:00 to 2026-11-19 12:00 checked; {len(bad)} start with only NUL bytes\n")
        fh.write("\n".join(bad) + ("\n" if bad else ""))
    print(f"{n} files checked; {len(bad)} zero-filled")
    print("\n".join(bad[:50]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
