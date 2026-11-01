#!/usr/bin/env python3
"""Split a parquet file into consecutive parts of whole row groups, rows kept in order.

weaver's --fetch-by-files loader holds one whole file in memory. The class-balanced pairs_test
(2.1 M jets) OOM-killed an extraction pod at 48Gi; pairs_train (1.1 M jets) fits. The parts, read
in order, give the same rows in the same order (checked on jc2bal_src), so extract_v2's stream
rows and pairs_index.npz are unchanged. The source file is kept.

    python3 experiments/FT/split_parquet.py /data/finetune/jc2_bal/pairs_test.parquet --parts 3
"""
import argparse
import pathlib
import sys

import numpy as np
import pyarrow.parquet as pq


def part_paths(src: pathlib.Path, parts: int) -> list:
    return [src.with_name(f"{src.stem}_part{i}.parquet") for i in range(parts)]


def split(src: pathlib.Path, parts: int, check: str = "jc2bal_src") -> list:
    f = pq.ParquetFile(src)
    out = part_paths(src, parts)
    for p, groups in zip(out, np.array_split(np.arange(f.num_row_groups), parts)):
        tmp = p.with_name(p.name + ".tmp")
        with pq.ParquetWriter(tmp, f.schema_arrow) as w:
            for g in groups:
                w.write_table(f.read_row_group(int(g)))
        tmp.rename(p)
    whole = pq.read_table(src, columns=[check]).column(check).to_numpy()
    cat = np.concatenate([pq.read_table(p, columns=[check]).column(check).to_numpy() for p in out])
    if not np.array_equal(whole, cat):
        raise SystemExit(f"FATAL: the parts of {src} do not reproduce its {check} column")
    print(f"{src}: {whole.size:,} rows -> " + ", ".join(f"{pq.ParquetFile(p).metadata.num_rows:,}" for p in out))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=pathlib.Path)
    ap.add_argument("--parts", type=int, required=True)
    a = ap.parse_args(argv)
    split(a.src, a.parts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
