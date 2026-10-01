"""v2 pretraining data streams for weaver 0.4.17: Sophon's loader, epoch-seeded.

WHAT v1 DID, AND WHY IT IS REPLACED (audit 2026-09-29, B1 and B4)
-----------------------------------------------------------------
v1 ran weaver 0.4.17's own loader with `--fetch-by-files --fetch-step 5
--num-workers 2`: each worker read 5 whole files of ONE family at a time
(weaver.utils.dataset._SimpleIter, files shuffled across families, then read in
blocks of 5), so the class mix of the stream moved with whichever family the
current block happened to hold. The epoch-level QCD share spanned 8.0-17.5%.
The workers were persistent and never reseeded, so the stream of epoch e was
"whatever came next", and a resume restarted it from the epoch-0 state.

Sophon (hqucms/weaver-core@c97de3c utils/dataset.py:203-239) reads with
`--fetch-step 1.0 --data-split-num 200`: every fetch loads the same fraction of
EVERY family's files (a family-stratified slice of about 1.3 files per worker),
so every fetch carries the nominal class mix. weaver 0.4.17 dropped
`--data-split-num`; with `--fetch-step 1.0` and no split it loads every file of
a worker in one fetch, which is what ran out of memory (CLAUDE.md section 8).

WHAT THIS MODULE DOES
---------------------
* `sophon_splits` is Sophon's schedule, ported line for line (n_div_d_sep).
* With --data-fraction F = 1/k (Sophon's flag), epoch e reads a window of every
  file: rows [j F, (j+1) F) of a per-cycle row permutation, j = e % k (cycle_of,
  file_rows), so every epoch samples every file and every row is read once per
  k epochs. Without it an epoch reads ~20% of the files, and which 20% moved the
  epoch's QCD share at 3.7x the binomial level (dry runs at mtx-s1.69/1.70).
* `StreamDataset(mode="train")` gives each epoch a FRESH stream. For worker w of
  epoch e, the file order within each family, the split schedule and therefore
  the load ranges, the reweighting draws and the row permutation of every fetch
  are functions of (data_sampling seed, e, w) and nothing else: each draw comes
  from its own numpy Generator seeded by SeedSequence(seed, spawn_key=(...)),
  never from a global RNG. So epoch e is the same stream whether or not the run
  was resumed before it, and whatever the vocabulary or output width.
* `StreamDataset(mode="val")` is the fixed validation sample: every row passing
  the selection, no reweighting and no shuffling, the same rows in the same
  order every epoch.
* Every batch carries each jet's identity (`_rowid` = file index * 2**20 + row
  in file), its native `jet_label`, its reweighting weight and the fetch it came
  from, in the third element of the (X, y, Z) batch. The training loop hashes
  `_rowid` into the per-epoch stream record (pretrain_v2.StreamRecord).

Reading is weaver 0.4.17's own: `_read_parquet` is `ak.from_parquet(path,
columns=...)` followed by a slice [trunc(lo * n), max(start + 1, trunc(hi * n))),
and selection, new variables, label check, weights and input finalisation call
weaver's functions. JetClass-II parquet files are ONE row group of 100,000 rows,
so a slice costs a whole-file read; `_FileCache` keeps the last file of each
family, which the next split of the same family continues from.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import awkward as ak
import numpy as np
import torch.utils.data

from weaver.utils.data.preprocess import _apply_selection, _build_new_variables, _build_weights
from weaver.utils.dataset import _check_labels, _finalize_inputs

try:  # weaver 0.4.17 (the image)
    from weaver.utils.data.tools import _get_variable_names
except ImportError:  # the dev branch installed on laptops
    from weaver.utils.data.eval_utils import _get_variable_names

ROW_BITS = 20                      # 100,000 rows per file < 2**20
TAG_FILES, TAG_FETCH, TAG_ROWS = 0, 1, 2   # SeedSequence spawn-key tags
NATIVE_QCD_FIRST = 161             # jet_label 161..187 are QCD (docs/GROUND_TRUTH.md)
N_NATIVE = 188


def seed_seq(seed: int, *key: int) -> np.random.SeedSequence:
    """The one derivation every data draw goes through."""
    return np.random.SeedSequence(int(seed), spawn_key=tuple(int(k) for k in key))


# ---------------------------------------------------------------- Sophon schedule
def n_div_d_sep(n: int, d: int) -> np.ndarray:
    """hqucms/weaver-core@c97de3c utils/dataset.py:210-214, verbatim.

    For n files and split number d, row di is the cumulative loaded fraction of
    each file after di splits: n=5, d=3 gives
    [[0,0,0,0,0], [1,2/3,0,0,0], [1,1,1,1/3,0], [1,1,1,1,1]].
    """
    return np.array([[np.clip(n * di / d - ni, 0, 1) for ni in range(n)] for di in range(d + 1)])


def sophon_splits(file_dict: dict, split_num: int, fetch_step: float = 1.0,
                  load_range=(0.0, 1.0)) -> list:
    """Sophon's load schedule (c97de3c utils/dataset.py:216-239), one worker.

    Returns [(files, ranges), ...]: one entry per fetch, each loading the same
    fraction of every family's files. `file_dict` is {family: [files in the
    order to read them]}; families are taken in sorted order (Sophon iterates a
    set; the order only fixes the concatenation before the row shuffle).
    """
    out = []
    lo, hi = load_range
    for i_load in range(math.ceil((hi - lo) / fetch_step)):
        start_pos = lo + i_load * fetch_step
        delta = min(fetch_step, hi - start_pos)
        splits = [([], []) for _ in range(split_num)]
        for name in sorted(file_dict):
            files = file_dict[name]
            n = len(files)
            arr = n_div_d_sep(n, split_num)
            for d in range(split_num):
                fs, rs = splits[d]
                for i in range(n):
                    if arr[d + 1, i] - arr[d, i] > 0:
                        fs.append(files[i])
                        rs.append((start_pos + delta * arr[d, i], start_pos + delta * arr[d + 1, i]))
        out += splits
    return out


def worker_files(file_dict: dict, worker: int, num_workers: int) -> dict:
    """weaver's split of the training files across workers: per family,
    sorted(files)[w::nw] (weaver 0.4.17 utils/dataset.py:147-152)."""
    out = {}
    for name, files in file_dict.items():
        mine = sorted(files)[worker::num_workers]
        if not mine:
            raise RuntimeError(f"family {name} has {len(files)} files, fewer than "
                               f"{num_workers} workers")
        out[name] = mine
    return out


def cycle_of(epoch: int, data_fraction: float):
    """(cycle, window) of an epoch. With data fraction F = 1/k, epoch e reads the
    rows at positions [j F, (j+1) F) of each file's row permutation for cycle
    e // k, j = e % k: every file every epoch, every row once per k epochs."""
    k = round(1.0 / data_fraction)
    if abs(k * data_fraction - 1.0) > 1e-9:
        raise ValueError(f"data fraction {data_fraction} is not 1/k")
    j = epoch % k
    return epoch // k, (j * data_fraction, (j + 1) * data_fraction if j + 1 < k else 1.0)


def window_of(epoch: int, windows: int):
    """(cycle, (j, k)) with an integer window count k: epoch e reads window j = e % k
    of cycle e // k, rows [j n // k, (j+1) n // k) of each file's permutation
    (file_rows), so the k windows of a cycle tile every file exactly, in integers."""
    k = int(windows)
    if k != windows or k < 1:
        raise ValueError(f"window count {windows} is not a positive integer")
    return epoch // k, (epoch % k, k)


def train_plan(file_dict: dict, seed: int, epoch: int, worker: int, num_workers: int,
               split_num: int, fetch_step: float, pass_idx: int = 0,
               data_fraction: float = 1.0, windows: int | None = None) -> list:
    """The fetch schedule of one worker in one pass of one epoch: Sophon's
    schedule over the epoch's window of every file (Sophon's --data-fraction).
    With an integer window count the load ranges are fractions OF THE WINDOW
    (file_rows maps them to rows); otherwise of the file (cycle_of)."""
    rng = np.random.default_rng(seed_seq(seed, TAG_FILES, epoch, worker, pass_idx))
    mine = worker_files(file_dict, worker, num_workers)
    shuffled = {name: [files[i] for i in rng.permutation(len(files))]
                for name, files in sorted(mine.items())}
    load_range = (0.0, 1.0) if windows else cycle_of(epoch, data_fraction)[1]
    return sophon_splits(shuffled, split_num, fetch_step, load_range=load_range)


def plan_sha256(file_dict: dict, seed: int, epoch: int, num_workers: int,
                split_num: int, fetch_step: float, data_fraction: float = 1.0,
                windows: int | None = None) -> str:
    """sha256 of every worker's first-pass schedule for the epoch: file base
    names and load ranges in read order, plus the worker count and split number
    that shape it (and, with an integer window count, the count and the epoch's
    window). An epoch that runs past its first pass is still covered by the row hash."""
    plan = {"num_workers": num_workers, "split_num": split_num, "fetch_step": fetch_step,
            "data_fraction": data_fraction,
            "workers": [[[[os.path.basename(f) for f in fs], [[round(a, 12), round(b, 12)] for a, b in rs]]
                         for fs, rs in train_plan(file_dict, seed, epoch, w, num_workers,
                                                  split_num, fetch_step, 0, data_fraction, windows)]
                        for w in range(num_workers)]}
    if windows:
        plan["windows"] = list(window_of(epoch, windows)[1])
    return hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()


# ---------------------------------------------------------------- reading
def _slice_bounds(n: int, lo: float, hi: float):
    """weaver 0.4.17 utils/data/fileio.py _read_parquet, the row slice."""
    start = math.trunc(lo * n)
    stop = max(start + 1, math.trunc(hi * n))
    return start, stop


def file_rows(n: int, lo: float, hi: float, seed: int, cycle: int, pass_idx: int,
              file_index: int, window=None) -> np.ndarray:
    """The rows a training load range (lo, hi) of a file takes: positions
    [trunc(lo*n), trunc(hi*n)) of a permutation of the file's rows fixed per
    (seed, cycle, pass, file), sorted. The splits of one epoch therefore tile
    that epoch's window of the file, and the k windows of a cycle tile the file,
    as Sophon's slices do, but each takes a random sample of the file's rows.

    Why not Sophon's contiguous slice: JetClass-II files are not shuffled
    inside. Measured 2026-09-29 on Res34P_0100, Res2P_0050 and QCD_0100, the
    mean jet p_T of consecutive 10,000-row chunks varies by 100-400 GeV, 20-40x
    its statistical error, so a slice carries the kinematics of whichever
    generation batch it falls in, and with them the reweighting acceptance.
    Random rows alone did not change the epoch scatter, though (dry run at
    mtx-s1.70, epoch by epoch within 5e-5 of mtx-s1.69): that comes from which
    files an epoch reads, fixed by the file order. Hence the window over every
    file each epoch (cycle_of, --data-fraction).

    window = (j, k), an integer window count: (lo, hi) are fractions of window j,
    positions [j n // k, (j+1) n // k) of the permutation, so the windows of a
    cycle tile the file in integer arithmetic (window_of)."""
    perm = np.random.default_rng(seed_seq(seed, TAG_ROWS, cycle, pass_idx, file_index)).permutation(n)
    if window is None:
        start, stop = _slice_bounds(n, lo, hi)
        return np.sort(perm[start:stop])
    j, k = window
    a, b = j * n // k, (j + 1) * n // k
    start, stop = _slice_bounds(b - a, lo, hi)
    return np.sort(perm[a + start:a + stop])


class _FileCache:
    """Whole-file reads (one row group per file), the last `size` kept."""

    def __init__(self, branches, size: int):
        self.branches = sorted(branches)
        self.size = max(1, int(size))
        self._c = OrderedDict()
        self.reads = 0

    def get(self, path: str):
        if path in self._c:
            self._c.move_to_end(path)
            return self._c[path]
        a = ak.from_parquet(path, columns=self.branches)
        self.reads += 1
        if len(a) >= 2 ** ROW_BITS:
            raise RuntimeError(f"{path}: {len(a)} rows does not fit the row id")
        self._c[path] = a
        while len(self._c) > self.size:
            self._c.popitem(last=False)
        return a


def _closure(data_config, names):
    """Raw branches and the ordered new-variable functions that `names` need."""
    funcs = data_config.var_funcs
    load, aux, todo = set(), set(), list(names)
    while todo:
        n = todo.pop()
        if n in funcs:
            if n not in aux:
                aux.add(n)
                todo.extend(_get_variable_names(funcs[n]))
        else:
            load.add(n)
    return load, {k: v for k, v in funcs.items() if k in aux}


def sidecar_path(path: str) -> str:
    """<config>.<md5 of config>.auto.yaml: the make_weight job's output, which
    carries the reweighting histograms (weaver utils/dataset.py:330-338)."""
    from weaver.utils.data.config import _md5
    return path.replace(".yaml", ".%s.auto.yaml" % _md5(path))


def sidecar_mismatch(config_path: str, side_path: str) -> list:
    """The top-level keys in which the sidecar is not its config plus reweighting
    histograms: both loaded through weaver's DataConfig as training loads them (same
    defaults; no observers, which weaver's make_weight drops when it writes the
    sidecar), compared in everything except weights.reweight_hists. A stale or
    renamed sidecar (another partition's labels, another selection) is non-empty."""
    from weaver.utils.data.config import DataConfig

    def opts(p):
        o = copy.deepcopy(DataConfig.load(p, load_observers=False).options)
        if isinstance(o.get("weights"), dict):
            o["weights"].pop("reweight_hists", None)
        return o
    a, b = opts(config_path), opts(side_path)
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


def load_config(path: str, extra_selection: str | None = None):
    """A training config as weaver loads it for training (no observers;
    --extra-selection ANDed onto the selection, train.py train_load)."""
    from weaver.utils.data.config import DataConfig
    return DataConfig.load(path, load_observers=False, extra_selection=extra_selection)


def native_to_class(data_config) -> np.ndarray:
    """The arm's class index of each native label 0..187, from its own config."""
    if data_config.label_type != "custom" or "truth_label" not in data_config.label_names:
        return None
    t = ak.Array({"jet_label": np.arange(N_NATIVE)})
    _, funcs = _closure(data_config, ["truth_label"])
    t = _build_new_variables(t, funcs)
    return ak.to_numpy(t["truth_label"]).astype(np.int64)


class StreamDataset(torch.utils.data.IterableDataset):
    """Training stream (mode="train") or fixed validation sample (mode="val").

    Yields whole batches (X, y, Z); use DataLoader(batch_size=None). With
    labels_only=True no particle column is read and X is empty: the loader
    dry run uses this, and draws exactly the rows training would draw, because
    the selection, the weights and every random draw see the same inputs.
    """

    def __init__(self, file_dict: dict, config_file: str, *, mode: str, batch_size: int,
                 seed: int | None = None, split_num: int = 200, fetch_step: float = 1.0,
                 labels_only: bool = False, max_resample: int = 10, cache_per_family: int = 1,
                 extra_selection: str | None = None, data_fraction: float = 1.0,
                 data_windows: int | None = None):
        self.config_file = str(config_file)
        self.extra_selection = extra_selection
        data_config = load_config(self.config_file, extra_selection)
        if mode not in ("train", "val"):
            raise ValueError(mode)
        if mode == "train" and seed is None:
            raise ValueError("the training stream needs its data_sampling seed")
        if data_config.options.get("file_magic") or data_config.options.get("branch_magic"):
            raise RuntimeError("file_magic / branch_magic are not supported")
        if data_config.weight_name is None or data_config.use_precomputed_weights:
            raise RuntimeError("v2 streams need on-the-fly reweighting histograms")
        if data_config.reweight_hists is None:
            raise RuntimeError("the data config carries no reweight_hists: copy its "
                               "*.auto.yaml sidecar next to it (never rebuilt here)")
        self.file_dict = {k: sorted(v) for k, v in file_dict.items()}
        self.mode = mode
        self.config = data_config
        self.batch_size = int(batch_size)
        self.seed = seed
        self.split_num = int(split_num)
        self.fetch_step = float(fetch_step)
        self.data_fraction = float(data_fraction)
        self.data_windows = data_windows
        if data_windows is None:
            cycle_of(0, self.data_fraction)      # refuses a fraction that is not 1/k
        elif data_fraction != 1.0:
            raise ValueError("data_windows replaces data_fraction")
        else:
            window_of(0, data_windows)           # refuses a count that is not a positive integer
        self.labels_only = labels_only
        self.max_resample = max_resample
        self.cache_size = cache_per_family * len(self.file_dict) + 1
        self.epoch = 0
        all_files = sorted(os.path.basename(f) for v in self.file_dict.values() for f in v)
        if len(set(all_files)) != len(all_files):
            raise RuntimeError("duplicate file base names: row ids would collide")
        self.file_index = {b: i for i, b in enumerate(all_files)}
        need = list(data_config.label_names) + list(data_config.reweight_branches) + \
            list(data_config.reweight_classes) + ["jet_label"]
        if labels_only:
            sel_vars = _get_variable_names(data_config.selection) if data_config.selection else []
            self.load_branches, self.aux_funcs = _closure(data_config, need + list(sel_vars))
        else:
            self.load_branches = set(data_config.train_load_branches) | {"jet_label"}
            self.aux_funcs = {k: v for k, v in data_config.var_funcs.items()
                              if k in data_config.train_aux_branches}

    # weaver's DataConfig does not survive pickling (its __getattr__ recurses),
    # so a spawned worker reloads it from the file instead.
    def __getstate__(self):
        st = dict(self.__dict__)
        st["config"] = None
        return st

    def __setstate__(self, st):
        self.__dict__.update(st)
        self.config = load_config(self.config_file, self.extra_selection)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    # -- one fetch
    def _load(self, cache, files, ranges, rng, key=None):
        """key = (cycle, pass, window) for the training stream: each file's load
        range then selects that many rows at random positions of the file
        (file_rows), not a contiguous slice. None (validation): contiguous."""
        parts = []
        for f, (lo, hi) in zip(files, ranges):
            full = cache.get(f)
            fidx = self.file_index[os.path.basename(f)]
            if key is None:
                start, stop = _slice_bounds(len(full), lo, hi)
                rows = np.arange(start, stop)
                part = full[start:stop]
            else:
                cycle, p, window = key
                rows = file_rows(len(full), lo, hi, self.seed, cycle, p, fidx, window)
                part = full[rows]
            parts.append(ak.with_field(part, fidx * 2 ** ROW_BITS + rows, "_rowid"))
        table = parts[0] if len(parts) == 1 else ak.concatenate(parts)
        cfg = self.config
        table = _apply_selection(table, cfg.selection, funcs=cfg.var_funcs)
        if len(table) == 0:
            return None
        table = _build_new_variables(table, self.aux_funcs)
        if cfg.label_type == "simple":
            _check_labels(table)
        wgts = _build_weights(table, cfg)
        if self.mode == "train":
            idx = reweight_indices(wgts, rng, max_resample=self.max_resample)
            rng.shuffle(idx)
        else:
            idx = np.arange(len(wgts))
        extra = {"_rowid": ak.to_numpy(table["_rowid"]).astype(np.int64),
                 "_jet_label": ak.to_numpy(table["jet_label"]).astype(np.int32),
                 "_weight": np.asarray(wgts, dtype=np.float32)}
        if self.labels_only:
            out = {k: ak.to_numpy(table[k]) for k in cfg.label_names}
        else:
            out = _finalize_inputs(table, cfg)
        out.update(extra)
        return out, idx

    def _fetches(self, worker, num_workers, cache):
        """(fetch id, files, ranges, rng) in read order for this worker."""
        if self.mode == "val":
            flat = sorted(f for v in self.file_dict.values() for f in v)
            for i, f in enumerate(flat[worker::num_workers]):
                yield i, [f], [(0.0, 1.0)], None, None
            return
        p = 0
        while True:  # a new pass only if an epoch outruns one (test-sized inputs)
            plan = train_plan(self.file_dict, self.seed, self.epoch, worker, num_workers,
                              self.split_num, self.fetch_step, p, self.data_fraction, self.data_windows)
            if self.data_windows:
                cycle, window = window_of(self.epoch, self.data_windows)
            else:
                cycle, window = cycle_of(self.epoch, self.data_fraction)[0], None
            for f, (files, ranges) in enumerate(plan):
                rng = np.random.default_rng(seed_seq(self.seed, TAG_FETCH, self.epoch, worker, p, f))
                yield p * len(plan) + f, files, ranges, rng, (cycle, p, window)
            p += 1

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        worker, num_workers = (wi.id, wi.num_workers) if wi is not None else (0, 1)
        cache = _FileCache(self.load_branches, self.cache_size)
        cfg = self.config
        xkeys = [] if self.labels_only else list(cfg.input_names)
        ykeys = list(cfg.label_names)
        zkeys = ["_rowid", "_jet_label", "_weight"]
        bs = self.batch_size
        pool = ThreadPoolExecutor(max_workers=1)   # overlap the next read with this one's batches
        fetches = self._fetches(worker, num_workers, cache)

        def submit():
            try:
                fid, files, ranges, rng, key = next(fetches)
            except StopIteration:
                return None
            return fid, pool.submit(self._load, cache, files, ranges, rng, key)

        def take(out, sel, fid):
            X = {k: out["_" + k][sel] for k in xkeys}
            y = {k: out[k][sel] for k in ykeys}
            Z = {k: out[k][sel] for k in zkeys}
            Z["_fetch"] = np.full(len(sel), fid, dtype=np.int32)
            return X, y, Z

        def cat(a, b):
            return tuple({k: np.concatenate([a[i][k], b[i][k]]) for k in a[i]} for i in range(3))

        try:
            nxt = submit()
            carry = None
            while nxt is not None:
                fid, fut = nxt
                res = fut.result()
                nxt = submit()
                if res is None:
                    continue
                out, idx = res
                pos = 0
                if carry is not None:
                    need = bs - len(carry[2]["_rowid"])
                    carry = cat(carry, take(out, idx[:need], fid))
                    pos = min(need, len(idx))
                    if len(carry[2]["_rowid"]) < bs:
                        continue
                    yield carry
                    carry = None
                while len(idx) - pos >= bs:
                    yield take(out, idx[pos:pos + bs], fid)
                    pos += bs
                if pos < len(idx):
                    carry = take(out, idx[pos:], fid)
            if carry is not None and self.mode == "val":
                yield carry          # validation keeps every row
        finally:
            pool.shutdown(wait=False, cancel_futures=True)


def reweight_indices(weights, rng, up_sample=True, max_resample=10, weight_scale=1):
    """weaver 0.4.17 utils/dataset.py:57-70 (_get_reweight_indices) with its two
    np.random.uniform draws taken from `rng` instead of the global RNG."""
    weights = np.asarray(weights)
    randwgt = rng.uniform(low=0, high=weight_scale, size=len(weights))
    keep_flags = randwgt < weights
    if not up_sample:
        return np.arange(len(weights))[keep_flags]
    n_repeats = len(weights) // max(1, int(keep_flags.sum()))
    if n_repeats > max_resample:
        n_repeats = max_resample
    all_indices = np.repeat(np.arange(len(weights)), n_repeats)
    randwgt = rng.uniform(low=0, high=weight_scale, size=len(weights) * n_repeats)
    return copy.deepcopy(all_indices[randwgt < np.repeat(weights, n_repeats)])
