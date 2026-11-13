"""Performance against jet pT and the JetClass confusion matrices (experiments/EVAL/perf_vs_pt.py) and
their job (scripts/build_perf_pt_jobs.py): the binning, the per-bin macro AUC against scikit-learn,
the X->bb vs X->cc rejection and its passing count, the row maps and their refusals, the confusion
matrix, and the job's guards and parts."""
import hashlib
import importlib.util
import json
import pathlib

import numpy as np
import pytest
import yaml

pytest.importorskip("sklearn")
from sklearn.metrics import roc_auc_score  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name, rel):
    s = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


PV = _load("perf_vs_pt", "experiments/EVAL/perf_vs_pt.py")
BJ = _load("build_perf_pt_jobs", "scripts/build_perf_pt_jobs.py")
JC1 = list(PV.LP.JC1_CLASSES)
SEL = "(jet_pt > 200) & (jet_pt < 2500) & (jet_sdmass > 20) & (jet_sdmass < 500)"


# ------------------------------------------------------------------ binning and metrics

def test_the_bins_are_the_atlas_ten_and_five_jetclass_bins_of_100_gev():
    ar = _load("atlas_rebin", "experiments/EVAL/atlas_rebin.py")
    assert PV.PT_EDGES["jc2"] == ar.PT_EDGES and len(PV.PT_EDGES["jc2"]) == 11
    assert PV.PT_EDGES["jc1"] == [500, 600, 700, 800, 900, 1000]


def test_a_bin_is_closed_below_open_above_and_the_last_edge_belongs_to_the_last_bin():
    b = PV.bin_index(np.array([199.9, 200, 299.99, 300, 2499, 2500, 2500.1]), PV.PT_EDGES["jc2"])
    assert b.tolist() == [-1, 0, 0, 1, 9, 9, -1]


def _sample(n=6000, k=10, seed=0):
    r = np.random.default_rng(seed)
    y = r.integers(0, k, n)
    p = r.dirichlet(np.ones(k), n)
    p[np.arange(n), y] += r.uniform(0, 1, n)
    return p / p.sum(1, keepdims=True), y, r.uniform(500, 1000, n)


def test_the_macro_auc_per_bin_is_scikit_learns_one_vs_rest_macro():
    p, y, pt = _sample()
    auc, k = PV.macro_auc(p, y)
    assert k == 10 and auc == pytest.approx(roc_auc_score(y, p, multi_class="ovr", average="macro"), abs=1e-12)
    doc = PV.binned(p, y, pt, PV.PT_EDGES["jc1"], PV.jc1_block)
    assert sum(b["n"] for b in doc["bins"]) + doc["n_outside"] == y.size and doc["n_outside"] == 0
    for b in doc["bins"]:
        m = (pt >= b["lo"]) & (pt < b["hi"])
        assert b["n"] == m.sum() and b["class_counts"] == np.bincount(y[m], minlength=10).tolist()
        assert b["macro_auc_ovr"] == pytest.approx(roc_auc_score(y[m], p[m], multi_class="ovr", average="macro"), abs=1e-12)
        # each signal class against QCD alone, on p_S / (p_S + p_QCD)
        c = 2
        mm = m & np.isin(y, (0, c))
        assert b["vs_qcd"][JC1[c]]["auc"] == pytest.approx(roc_auc_score(y[mm] == c, p[mm, c] / (p[mm, c] + p[mm, 0])))
    assert doc["bins"][0]["vs_qcd"]["label_Hqql"]["eps_s"] == 0.99 and doc["bins"][0]["vs_qcd"]["label_Tbl"]["eps_s"] == 0.995
    # a class missing from a bin leaves the mean over the classes present
    m = y != 3
    assert PV.macro_auc(p[m], y[m])[1] == 9


def test_the_accuracy_per_bin_is_the_argmax_share_with_its_binomial_spread():
    p, y, pt = _sample()
    for block in (PV.jc1_block, PV.jc2_block(0, 1)):
        doc = PV.binned(p, y, pt, PV.PT_EDGES["jc1"], block)
        for b in doc["bins"] + [doc["inclusive"]]:
            m = (pt >= b["lo"]) & (pt < b["hi"]) if "lo" in b else np.ones_like(y, bool)
            acc = float(np.mean(p[m].argmax(1) == y[m]))
            assert b["accuracy"] == pytest.approx(acc, abs=1e-12)
            assert b["accuracy_stat"] == pytest.approx(np.sqrt(acc * (1 - acc) / m.sum()), abs=1e-12)
    assert PV.accuracy(p[:0], y[:0]) == {"accuracy": None, "accuracy_stat": None}


def test_bb_vs_cc_rejection_carries_its_passing_count():
    n = 1000
    d_sig = 0.5 + np.arange(1, n + 1) / (2 * n)              # signal median 0.75
    d_bkg = np.r_[np.full(40, 0.9001), np.full(960, 0.1)]     # 40 background jets above every threshold near the median
    d = np.r_[d_sig, d_bkg]
    p = np.stack([d, 1 - d, np.zeros_like(d)], axis=1)
    y = np.r_[np.zeros(n, int), np.ones(n, int)]
    r = PV.versus(p, y, 0, (1,), 0.5)
    assert r["eps_b"] == pytest.approx(0.04) and r["n_bkg_pass"] == 40 and r["rejection"] == pytest.approx(25.0)
    assert r["rel_stat"] == pytest.approx(1 / np.sqrt(40)) and not r["is_bound"]
    assert r["n_signal"] == n and r["n_background"] == n and r["auc"] == pytest.approx(roc_auc_score(y == 0, d))
    # no background jet passes: the rejection is capped at the count and flagged
    q = np.stack([np.r_[d_sig, np.full(n, 0.1)], 1 - np.r_[d_sig, np.full(n, 0.1)], np.zeros(2 * n)], axis=1)
    r = PV.versus(q, y, 0, (1,), 0.5)
    assert r["is_bound"] and r["n_bkg_pass"] == 0 and r["rel_stat"] is None and r["rejection"] == n
    # a bin without background jets has no comparison
    assert PV.versus(p[:n], y[:n], 0, (1,), 0.5) is None


def test_the_confusion_matrix_is_counts_and_rows_normalised_to_one():
    y = np.array([0, 0, 1, 2, 2, 2])
    pred = np.array([0, 1, 1, 2, 2, 0])
    p = np.eye(4)[pred] * 0.7 + 0.075
    c = PV.confusion(p, y)
    assert c["counts"] == [[1, 1, 0, 0], [0, 1, 0, 0], [1, 0, 2, 0], [0, 0, 0, 0]]
    assert c["row_normalised"][:3] == [[0.5, 0.5, 0, 0], [0, 1, 0, 0], [pytest.approx(1 / 3), 0, pytest.approx(2 / 3), 0]]
    assert c["row_normalised"][3] is None
    p, y, _ = _sample()
    c = PV.confusion(p, y)
    assert np.sum(c["counts"]) == y.size and np.allclose(np.sum(c["row_normalised"], axis=1), 1.0)


# ------------------------------------------------------------------ JetClass-II: the row map

def _jc2_inputs(tmp, n=3000, seed=1):
    pa, pq = pytest.importorskip("pyarrow"), pytest.importorskip("pyarrow.parquet")
    r = np.random.default_rng(seed)
    lab = r.integers(0, 10, n + 50)
    pt = r.uniform(200.5, 2499, n + 50).astype(np.float32)
    msd = r.uniform(25, 480, n + 50).astype(np.float32)
    msd[r.choice(n + 50, 50, replace=False)] = 10.0          # fail the selection: not in the stream
    pq.write_table(pa.table({"jet_label": lab, "jet_pt": pt, "jet_sdmass": msd}), tmp / "test.parquet")
    keep = msd > 20
    y, ptk = lab[keep], pt[keep].astype(np.float64)
    cfg = tmp / "cfg.yaml"
    cfg.write_text(f"selection:\n   {SEL}\n")
    p = r.dirichlet(np.ones(188), y.size)
    p[np.arange(y.size), y] += 0.05
    p /= p.sum(1, keepdims=True)
    d = tmp / "probs" / "m" / "best70"
    d.mkdir(parents=True)
    np.savez(d / "jc2bal_N1000000.npz", rows=np.arange(y.size), y=y.astype(np.int16), p=p.astype(np.float16))
    e = tmp / "ext" / "m" / "jc2bal" / "test" / "best70"
    e.mkdir(parents=True)
    np.save(e / "label188.npy", y.astype(np.int16))
    (e / "manifest.json").write_text(json.dumps({"data_config": str(cfg), "n_stream": int(y.size),
                                                 "data_config_sha256": hashlib.sha256(cfg.read_bytes()).hexdigest()}))
    return y, ptk, p.astype(np.float16).astype(np.float32)


def _jc2_args(tmp, out):
    return ["jc2", "--out-dir", str(out), "--models", "m", "--probs-root", str(tmp / "probs"),
            "--extract-root", str(tmp / "ext"), "--test-parquet", str(tmp / "test.parquet")]


def test_jc2_bins_the_probe_on_the_selected_rows_of_the_test_file(tmp_path):
    y, pt, p = _jc2_inputs(tmp_path)
    out = tmp_path / "out"
    assert PV.main(_jc2_args(tmp_path, out)) == 0
    doc = json.loads((out / "m.json").read_text())
    assert doc["pt_edges"] == PV.PT_EDGES["jc2"] and len(doc["bins"]) == 10 and doc["n_jets"] == y.size
    assert sum(b["n"] for b in doc["bins"]) == y.size and doc["jets_sha256"] == PV.jets_sha(y, pt)
    b = doc["bins"][3]
    m = (pt >= b["lo"]) & (pt < b["hi"])
    assert b["macro_auc_ovr"] == pytest.approx(np.mean([roc_auc_score(y[m] == c, p[m, c]) for c in np.unique(y[m])]))
    assert b["n_bb"] == (y[m] == 0).sum() and b["bb_vs_cc"]["n_background"] == (y[m] == 1).sum()
    assert not list(out.glob("*.new"))
    # an existing file is skipped, not rewritten
    t = (out / "m.json").stat().st_mtime_ns
    assert PV.main(_jc2_args(tmp_path, out)) == 0 and (out / "m.json").stat().st_mtime_ns == t


def test_jc2_refuses_a_test_file_whose_rows_are_not_the_streams(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    _jc2_inputs(tmp_path)
    t = pq.read_table(tmp_path / "test.parquet")
    pq.write_table(t.take(np.arange(t.num_rows)[::-1]), tmp_path / "test.parquet")
    with pytest.raises(SystemExit, match="not the same jets"):
        PV.main(_jc2_args(tmp_path, tmp_path / "out"))
    assert not (tmp_path / "out" / "m.json").exists()


def test_jc2_refuses_a_data_config_that_is_not_the_extractions(tmp_path):
    _jc2_inputs(tmp_path)
    (tmp_path / "cfg.yaml").write_text(f"selection:\n   {SEL} & (jet_pt > 0)\n")
    with pytest.raises(SystemExit, match="not the data config"):
        PV.main(_jc2_args(tmp_path, tmp_path / "out"))


# ------------------------------------------------------------------ JetClass: weaver's predict order

def test_the_loader_takes_one_batch_from_each_worker_in_turn():
    pos = PV.interleave([5, 3], 2)
    assert pos[0].tolist() == [0, 1, 4, 5, 7] and pos[1].tolist() == [2, 3, 6]


def _log(workers):
    return "".join(f"[2026-11-05 10:00:0{w}] INFO: Restarted DataIter test__worker{w}, load_range=(0, 1), file_list:\n"
                   + json.dumps({"": fs}, indent=2) + "\n" for w, fs in enumerate(workers))


def test_the_worker_file_lists_are_read_from_predict_log(tmp_path):
    f = tmp_path / "predict.log"
    f.write_text("[x] INFO: args\n" + _log([["a", "c"], ["b", "d"]]) + _log([["a", "c"]]) + "done\n")
    assert PV.predict_workers(f) == [["a", "c"], ["b", "d"]]
    f.write_text(_log([["a", "c"], ["b", "d"]]).replace("worker0", "worker2"))
    with pytest.raises(SystemExit, match="row order is unknown"):
        PV.predict_workers(f)
    f.write_text("no worker records\n")
    with pytest.raises(SystemExit, match="row order is unknown"):
        PV.predict_workers(f)


def _jc1_tree(tmp, uproot, per_file=700, swap=False):
    """Four test files (QCD and H->bb, two each), two workers, and one long-budget cell whose
    pred.root is in weaver's order; swap writes a predict.log whose workers hold one class each."""
    r = np.random.default_rng(3)
    files, data = {}, {}
    for cls in (0, 1):
        for half in "ab":
            name = str(tmp / f"{JC1[cls]}_{half}.root")
            pt = r.uniform(500, 1000, per_file).astype(np.float32)
            try:
                with uproot.recreate(name) as f:
                    f["tree"] = {**{c: np.full(per_file, i == cls) for i, c in enumerate(JC1)}, "jet_pt": pt}
            except AttributeError as exc:        # uproot 5.1 cannot write under numpy 2
                pytest.skip(f"this uproot cannot write ROOT files here: {exc}")
            files[(cls, half)], data[name] = name, (np.full(per_file, cls), pt.astype(np.float64))
    workers = [[files[(0, "a")], files[(1, "a")]], [files[(0, "b")], files[(1, "b")]]]
    # the stream, written out by hand: batches of 512, worker 0 first
    w = [(np.r_[data[a][0], data[b][0]], np.r_[data[a][1], data[b][1]]) for a, b in workers]
    cuts = [(0, 512), (512, 1024), (1024, 2 * per_file)]
    y = np.concatenate([w[k][0][lo:hi] for lo, hi in cuts for k in (0, 1)])
    pt = np.concatenate([w[k][1][lo:hi] for lo, hi in cuts for k in (0, 1)])
    p = r.dirichlet(np.ones(10), y.size)
    p[np.arange(y.size), y] += r.uniform(0, 1, y.size)
    cell = tmp / "tree" / "l188-s1" / "N1000000" / "s1"
    cell.mkdir(parents=True)
    with uproot.recreate(cell / "pred.root") as f:
        f["Events"] = {**{c: y == i for i, c in enumerate(JC1)},
                       **{f"score_{c}": p[:, i].astype(np.float32) for i, c in enumerate(JC1)}}
    log_workers = [[files[(0, "a")], files[(0, "b")]], [files[(1, "a")], files[(1, "b")]]] if swap else workers
    (cell / "predict.log").write_text(_log(log_workers))
    (cell / "DONE").touch()
    return y, pt, p


def _jc1_args(tmp):
    return ["jc1", "--tree", str(tmp / "tree"), "--out-dir", str(tmp / "out"), "--confusion-out", str(tmp / "conf")]


def test_jc1_rebuilds_pt_in_weavers_order_and_writes_the_confusion_matrix(tmp_path):
    uproot = pytest.importorskip("uproot")
    y, pt, p = _jc1_tree(tmp_path, uproot)
    assert PV.main(_jc1_args(tmp_path)) == 0
    doc = json.loads((tmp_path / "out" / "mtx-l188-s1.json").read_text())
    assert doc["pt_edges"] == PV.PT_EDGES["jc1"] and doc["jets_sha256"] == PV.jets_sha(y, pt)
    assert [b["n"] for b in doc["bins"]] == [int(((pt >= lo) & (pt < lo + 100)).sum()) for lo in range(500, 1000, 100)]
    assert doc["bins"][0]["vs_qcd"]["label_Hbb"]["n_background"] == int(((pt < 600) & (y == 0)).sum())
    conf = json.loads((tmp_path / "conf" / "mtx-l188-s1.json").read_text())
    pn = p.astype(np.float32).astype(np.float64)       # the scores as pred.root stores them
    assert conf["classes"] == JC1 and conf["counts"] == PV.confusion(pn, y)["counts"]
    assert np.sum(conf["counts"]) == y.size and conf["row_normalised"][2] is None
    assert np.allclose([sum(r) for r in conf["row_normalised"][:2]], 1.0)


def test_jc1_refuses_a_predict_order_whose_labels_differ_from_pred_root(tmp_path):
    uproot = pytest.importorskip("uproot")
    _jc1_tree(tmp_path, uproot, swap=True)
    with pytest.raises(SystemExit, match="predict.log's order"):
        PV.main(_jc1_args(tmp_path))
    assert not list((tmp_path / "out").glob("*.json"))


def test_a_model_scored_on_other_jets_than_the_written_ones_is_refused(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps({"jets_sha256": "x"}))
    with pytest.raises(SystemExit, match="not scored on the jets"):
        PV.same_jets(tmp_path, "y", "m")
    PV.same_jets(tmp_path, "x", "m")


# ------------------------------------------------------------------ round 5: JetClass-II fine-tuned, JetClass probed

def _jc2ft_cell(cell: pathlib.Path, y, pt, logits):
    fd = cell / "features"
    fd.mkdir(parents=True)
    np.save(fd / "logits.npy", logits.astype(np.float32))
    np.save(fd / "label188.npy", y.astype(np.int16))
    np.savez(fd / "observers.npz", jet_pt=pt.astype(np.float32), jet_energy=pt.astype(np.float32))
    (cell / "DONE").touch()


def test_jc2ft_bins_the_reported_cell_with_the_pt_of_its_own_extraction(tmp_path):
    r = np.random.default_rng(4)
    y = r.integers(0, 188, 4000)
    pt = r.uniform(200.5, 2499, y.size).astype(np.float32)
    logits = r.normal(size=(y.size, 188))
    logits[np.arange(y.size), y] += 3
    long_, std, scr = tmp_path / "long", tmp_path / "std", tmp_path / "scratch"
    _jc2ft_cell(std / "l188-s1" / "N1000000" / "s1", y, pt, logits - 1)    # superseded by the long cell
    _jc2ft_cell(long_ / "l188-s1" / "N1000000" / "s1", y, pt, logits)
    _jc2ft_cell(std / "l162-s1" / "N1000000" / "s1", y, pt, logits)
    _jc2ft_cell(scr / "scratch" / "N1000000" / "s2", y, pt, logits)
    out = tmp_path / "out"
    args = ["jc2ft", "--out-dir", str(out), "--trees", str(long_), str(std), "--scratch-tree", str(scr)]
    assert PV.main(args + ["--models", "mtx-l188-s1", "mtx-l162-s1", "scratch-s2"]) == 0
    doc = json.loads((out / "mtx-l188-s1.json").read_text())
    assert doc["cell"] == str(long_ / "l188-s1" / "N1000000" / "s1") and doc["mode"] == "fine-tuned"
    assert doc["pt_edges"] == PV.PT_EDGES["jc2"] and doc["jets_sha256"] == PV.jets_sha(y, pt.astype(np.float64))
    assert doc["same_jets_as_probe"] is None and set(doc["bins"][0]) == set(_probe_bin_keys())
    p = np.exp(logits - logits.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    m = (pt >= doc["bins"][2]["lo"]) & (pt < doc["bins"][2]["hi"])
    assert doc["bins"][2]["accuracy"] == pytest.approx(np.mean(p[m].argmax(1) == y[m]))
    assert json.loads((out / "scratch-s2.json").read_text())["cell"] == str(scr / "scratch" / "N1000000" / "s2")
    with pytest.raises(SystemExit, match="no finished JetClass-II cell"):
        PV.main(args + ["--models", "mtx-r42q1-s1"])
    (std / "l162-s1" / "N1000000" / "s1" / "train.log").write_text("AvgLoss: nan\n")
    (out / "mtx-l162-s1.json").unlink()
    with pytest.raises(SystemExit, match="NaN"):
        PV.main(args + ["--models", "mtx-l162-s1"])


def _probe_bin_keys():
    return ["lo", "hi", "n", "accuracy", "accuracy_stat", "macro_auc_ovr", "n_classes_present", "class_counts",
            "bb_vs_cc", "n_bb", "n_cc"]


def _jc1probe(tmp, uproot, per_file=400, model="mtx-l188-s1", ck="best70"):
    """Four JetClass test files (QCD and H->bb, two each), one probe extraction of them in file order, and the
    probe's outputs on every 2nd row."""
    tmp.mkdir(parents=True, exist_ok=True)
    r = np.random.default_rng(8)
    files, ys, pts = [], [], []
    for cls in (0, 1):
        for half in "ab":
            name = str(tmp / f"{JC1[cls]}_{half}.root")
            pt = r.uniform(500, 1000, per_file).astype(np.float32)
            try:
                with uproot.recreate(name) as f:
                    f["tree"] = {**{c: np.full(per_file, i == cls) for i, c in enumerate(JC1)}, "jet_pt": pt}
            except AttributeError as exc:
                pytest.skip(f"this uproot cannot write ROOT files here: {exc}")
            files.append(name), ys.append(np.full(per_file, cls)), pts.append(pt.astype(np.float64))
    y, pt = np.concatenate(ys), np.concatenate(pts)
    cfg = tmp / "jc1.yaml"
    cfg.write_text("selection:\n")
    e = tmp / "ext" / model / "jc1" / "test" / ck
    e.mkdir(parents=True)
    np.save(e / "label188.npy", y.astype(np.int16))
    np.save(e / "rows.npy", np.arange(y.size))
    (e / "manifest.json").write_text(json.dumps({"data_config": str(cfg), "n_stream": int(y.size), "n_test_files": 4,
                                                 "data_config_sha256": hashlib.sha256(cfg.read_bytes()).hexdigest()}))
    rows = np.arange(0, y.size, 2)
    p = r.dirichlet(np.ones(10), rows.size)
    p[np.arange(rows.size), y[rows]] += 0.5
    p /= p.sum(1, keepdims=True)
    d = tmp / "probs" / model / ck
    d.mkdir(parents=True)
    np.savez(d / "jc1_N1000000.npz", rows=rows, y=y[rows].astype(np.int16), p=p.astype(np.float16))
    return files, y[rows], pt[rows]


def test_jc1probe_takes_pt_from_the_test_files_in_the_extractions_order(tmp_path):
    uproot = pytest.importorskip("uproot")
    files, y, pt = _jc1probe(tmp_path, uproot)
    _jc1probe(tmp_path / "i", uproot, model="init-s1", ck="init")
    out = tmp_path / "out"
    base = ["jc1probe", "--out-dir", str(out), "--test-files", *files]
    assert PV.main(base + ["--models", "mtx-l188-s1", "--probs-root", str(tmp_path / "probs"),
                           "--extract-root", str(tmp_path / "ext")]) == 0
    doc = json.loads((out / "mtx-l188-s1.json").read_text())
    assert doc["checkpoint"] == "best70" and doc["pt_edges"] == PV.PT_EDGES["jc1"] and doc["n_jets"] == y.size
    assert doc["jets_sha256"] == PV.jets_sha(y, pt) and set(doc["bins"][0]["vs_qcd"]) == set(JC1[1:])
    assert PV.main(["jc1probe", "--out-dir", str(out), "--test-files", *files, "--models", "init-s1",
                    "--probs-root", str(tmp_path / "i" / "probs"), "--extract-root", str(tmp_path / "i" / "ext")]) == 0
    assert json.loads((out / "init-s1.json").read_text())["checkpoint"] == "init"
    # the random network with BatchNorm recomputed: folder init-s1, checkpoint init_bn, written as initbn-s1
    _jc1probe(tmp_path / "b", uproot, model="init-s1", ck="init_bn")
    assert PV.main(["jc1probe", "--out-dir", str(out), "--test-files", *files, "--models", "initbn-s1",
                    "--probs-root", str(tmp_path / "b" / "probs"), "--extract-root", str(tmp_path / "b" / "ext")]) == 0
    assert json.loads((out / "initbn-s1.json").read_text())["checkpoint"] == "init_bn"
    assert PV.INIT_BN_MODELS == ("initbn-s1", "initbn-s2", "initbn-s3")
    # the files in another order are other jets
    with pytest.raises(SystemExit, match="not the same jets"):
        PV.main(["jc1probe", "--out-dir", str(tmp_path / "o2"), "--test-files", *files[2:], *files[:2],
                 "--models", "mtx-l188-s1", "--probs-root", str(tmp_path / "probs"), "--extract-root", str(tmp_path / "ext")])
    with pytest.raises(SystemExit, match="read 4 test files"):
        PV.main(["jc1probe", "--out-dir", str(tmp_path / "o3"), "--test-files", *files[:3],
                 "--models", "mtx-l188-s1", "--probs-root", str(tmp_path / "probs"), "--extract-root", str(tmp_path / "ext")])

# ------------------------------------------------------------------ the job

def _job(parts):
    d = yaml.safe_load(BJ.spec("mtx-s9.99", parts))
    c = d["spec"]["template"]["spec"]["containers"][0]
    return d, c, c["args"][0]


def test_the_job_is_one_cpu_pod_with_the_guards():
    for parts in (["jc2"], ["jc1"], ["jc2", "jc1"]):
        d, c, s = _job(parts)
        assert d["metadata"]["name"] == BJ.name(parts) and d["metadata"]["name"].endswith("-raunav")
        assert d["spec"]["backoffLimit"] == 1 and "completionMode" not in d["spec"]
        assert d["spec"]["podFailurePolicy"]["rules"][0] == {
            "action": "FailJob", "onExitCodes": {"containerName": "main", "operator": "In", "values": [42]}}
        assert "nvidia.com/gpu" not in c["resources"]["limits"]
        assert d["spec"]["template"]["spec"]["volumes"][0]["persistentVolumeClaim"]["claimName"] == "transfer-learning-vol"
        term = d["spec"]["template"]["spec"]["affinity"]["nodeAffinity"][
            "requiredDuringSchedulingIgnoredDuringExecution"]["nodeSelectorTerms"][0]["matchExpressions"]
        assert {"key": "topology.kubernetes.io/region", "operator": "In", "values": ["us-west"]} in term
        assert "--branch mtx-s9.99" in s and "retry clone || exit 1" in s
        guard = s.index('[ "${USED}" -le 85 ] || { echo "FATAL: /data is ${USED}% full"; exit 42; }')
        assert guard < s.index("python3 experiments/EVAL/perf_vs_pt.py")


def test_the_jetclass_part_waits_for_all_30_long_cells_and_the_jetclass_ii_part_does_not():
    cells = BJ.long_jc1_cells()
    assert len(cells) == 30 and len(set(cells)) == 30
    assert sum(c.startswith("/data/results/ft_v2/budget_long/leg2/scratch/N1000000/s") for c in cells) == 3
    _, _, s = _job(["jc2"])
    assert "perf_vs_pt.py jc2 --out-dir /data/results/eval/perf_vs_pt_v2/jc2" in s
    assert "not DONE" not in s and "perf_vs_pt.py jc1" not in s and "pyarrow==25.0.1" in s
    _, _, s = _job(["jc1"])
    assert all(c in s for c in cells) and 'not DONE"; exit 42; }' in s
    assert s.index("not DONE") < s.index(
        "perf_vs_pt.py jc1 --out-dir /data/results/eval/perf_vs_pt_v2/jc1 --confusion-out /data/results/eval/confusion_jc1")
    assert "perf_vs_pt.py jc2" not in s
    # both: nothing runs before every JetClass cell is DONE
    _, _, s = _job(["jc2", "jc1"])
    assert s.index("not DONE") < s.index("perf_vs_pt.py jc2") < s.index("perf_vs_pt.py jc1")


def test_the_round5_parts_clone_a_commit_and_write_new_folders():
    d = yaml.safe_load(BJ.spec("ab" * 20, ["jc2ft", "jc1probe", "jc1full"], True, 89))
    s = d["spec"]["template"]["spec"]["containers"][0]["args"][0]
    assert d["metadata"]["name"] == "perf-pt-jc2ft-jc1probe-jc1full-raunav" and d["spec"]["backoffLimit"] == 1
    assert f"git -C tls checkout -q {'ab' * 20}" in s and "--branch" not in s
    assert '[ "${USED}" -le 89 ]' in s and "configmap" not in s.lower()
    assert f"perf_vs_pt.py jc2ft --out-dir {BJ.OUT_V3}/jc2_ft --probe-dir {BJ.OUT_ROOT}/jc2" in s
    assert f"perf_vs_pt.py jc1probe --out-dir {BJ.OUT_V3}/jc1_probe --test-files ${{TEST1}}" in s
    assert f"perf_vs_pt.py jc1 --tree /data/results/ft_v2/budget_long_fulldecay/leg2 --out-dir {BJ.OUT_V3}/jc1_fulldecay" in s
    assert BJ.CONF_FULL != BJ.CONF_ROOT and BJ.OUT_V3 != BJ.OUT_ROOT       # nothing written next to the old files
    # the 20 test files exactly as the probe extraction listed them (build_linprobe_jobs.LIST_TEST1)
    LPJ = _load("build_linprobe_jobs", "scripts/build_linprobe_jobs.py")
    assert LPJ.BF.JC1_CLASSES.split() == list(BJ.BFB.JC1_CLASSES)
    assert "| sort | head -2" in s and "-eq 20 ]" in s
    cells = BJ.full_jc1_cells()
    assert len(cells) == 3 and all(c in s for c in cells) and s.index("not DONE") < s.index("perf_vs_pt.py")
