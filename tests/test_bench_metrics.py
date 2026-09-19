"""Top-tagging and quark/gluon benchmark readout. The defects these pin are the
ones that would produce a plausible-looking community-table row rather than an
error: an inverted signal column, a rejection at the 1/N_bkg cap printed as a
measurement, and a mean over cells that did not read out the same jets."""
import importlib.util
import json
import pathlib

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


M = _load("bench_metrics", "experiments/FT/bench_metrics.py")
PROBE = _load("probe", "experiments/EVAL/probe.py")


def _cell(root, y, z, dataset="top", init="l162-s1b", n="N1000", seed="s1",
          *, done=True, subset=None):
    """One LEGS_BENCH cell. `z` is the signal-minus-background logit margin, or
    a full logits matrix."""
    cell = root / f"leg_{dataset}" / init / n / seed
    fd = cell / "features"
    fd.mkdir(parents=True)
    z = np.asarray(z, dtype=np.float32)
    if z.ndim == 1:
        z = np.stack([np.zeros_like(z), z], axis=1)
    np.save(fd / "label188.npy", np.asarray(y).astype(np.int16))
    np.save(fd / "logits.npy", z)
    if subset:
        (cell / "ft_manifest.json").write_text(json.dumps({"subset": subset}))
    if done:
        (cell / "DONE").touch()
    return cell


def _separable(n=4000, sep=2.0, seed=0):
    rng = np.random.default_rng(seed)
    y = np.tile([0, 1], n // 2)
    return y, rng.normal(0.0, 1.0, n) + sep * y


def _strict(path):
    """Parse as STRICT JSON: a bare Infinity or NaN is a failure, not a float."""
    def refuse(tok):
        raise AssertionError(f"non-JSON constant {tok} in {path}")
    return json.loads(path.read_text(), parse_constant=refuse)


def test_the_committed_data_configs_still_put_the_signal_in_column_one():
    """logits.npy carries no names, so column 1 = top / quark rests entirely on
    `value: [<background>, <signal>]` in the two data configs. Reversed, every
    AUC becomes 1 - AUC and still looks like a number."""
    for d in M.SIGNAL:
        M.check_label_order(d)
    assert M.SIGNAL["top"][1] == ["label_QCD", "label_Top"]
    assert M.SIGNAL["qg"][1] == ["label_gluon", "label_quark"]


def test_a_reordered_data_config_is_fatal(tmp_path, monkeypatch):
    cfg = tmp_path / "configs/finetune/TopReference.yaml"
    cfg.parent.mkdir(parents=True)
    text = (REPO / M.SIGNAL["top"][0]).read_text()
    cfg.write_text(text.replace("[label_QCD, label_Top]", "[label_Top, label_QCD]"))
    monkeypatch.setattr(M, "REPO", tmp_path)
    with pytest.raises(SystemExit, match="declares labels"):
        M.check_label_order("top")


def test_perfect_separation_is_auc_one_and_the_rejection_is_a_flagged_bound(tmp_path):
    """Zero background jets pass, so 1/eps_B is the 1/N_bkg cap -- a lower bound
    set by the sample size. It must carry its flag and no relative error."""
    y = np.tile([0, 1], 500)
    m = M.cell_metrics(_cell(tmp_path, y, np.where(y == 1, 5.0, -5.0)), PROBE)
    assert m["auc"] == 1.0 and m["accuracy"] == 1.0
    assert m["log1m_auc_censored"] is True
    for k in ("r50", "r30"):
        assert m[f"{k}_is_bound"] is True
        assert m[k] == 500.0 and m[f"{k}_n_bkg_pass"] == 0
        assert m[f"{k}_eps_b"] == 0.0 and m[f"{k}_rel_stat"] is None
        assert "BOUND" in M._rej(m, k) and ">=" in M._rej(m, k)
    assert (m["n_signal"], m["n_background"], m["n_jets"]) == (500, 500, 1000)


def test_random_scores_give_auc_half_and_the_trivial_rejections(tmp_path):
    """A score carrying no information has eps_B = eps_S, so R50 = 2 and
    R30 = 1/0.3. An inverted or mis-thresholded readout does not land here."""
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200_000)
    m = M.cell_metrics(_cell(tmp_path, y, rng.normal(size=y.size)), PROBE)
    assert m["auc"] == pytest.approx(0.5, abs=0.005)
    assert m["accuracy"] == pytest.approx(0.5, abs=0.005)
    assert m["r50"] == pytest.approx(2.0, abs=0.03)
    assert m["r30"] == pytest.approx(1 / 0.3, abs=0.07)
    assert not m["r50_is_bound"] and not m["r30_is_bound"]


def test_a_constructed_roc_gives_the_exact_rejection_at_each_working_point(tmp_path):
    """1000 background, 10 signal. 4 background jets outrank every signal jet,
    6 more sit between the 4th and 5th signal jet, the rest below all of them.
    So eps_B is exactly 4/1000 for eps_S in (0, 0.4) and 10/1000 in (0.4, 1):
    R30 = 250 and R50 = 100, and the two working points cannot be swapped."""
    sig = np.linspace(60, 51, 10)                    # descending signal margins
    bkg = np.concatenate([np.full(4, 99.0), np.full(6, 56.5), np.full(990, 1.0)])
    y = np.concatenate([np.ones(10), np.zeros(1000)]).astype(int)
    z = np.concatenate([sig, bkg]) / 10.0            # keep softmax unsaturated
    m = M.cell_metrics(_cell(tmp_path, y, z), PROBE)
    assert m["n_score_saturated"] == 0
    assert m["r30_eps_b"] == pytest.approx(0.004, rel=1e-12)
    assert m["r30"] == pytest.approx(250.0, rel=1e-12)
    assert m["r30_n_bkg_pass"] == 4 and m["r30_rel_stat"] == pytest.approx(0.5)
    assert m["r50"] == pytest.approx(100.0, rel=1e-12)
    assert m["r50_n_bkg_pass"] == 10
    assert not m["r30_is_bound"] and not m["r50_is_bound"]


def test_the_signal_column_is_column_one_not_column_zero(tmp_path):
    """The same margins written into column 0 must give 1 - AUC."""
    y, s = _separable()
    good = M.cell_metrics(_cell(tmp_path / "a", y, s), PROBE)
    flipped = np.stack([s, np.zeros_like(s)], axis=1)
    bad = M.cell_metrics(_cell(tmp_path / "b", y, flipped), PROBE)
    assert good["auc"] > 0.9
    assert bad["auc"] == pytest.approx(1.0 - good["auc"], abs=1e-9)


def test_three_column_logits_are_fatal(tmp_path):
    """argmax and a column-1 slice both still work on a 3-wide head, so a cache
    from another leg pointed at this reader would produce numbers."""
    y = np.tile([0, 1], 50)
    cell = _cell(tmp_path, y, np.zeros((100, 3)))
    with pytest.raises(SystemExit, match=r"not \(n, 2\)"):
        M.cell_metrics(cell, PROBE)


def test_labels_outside_zero_one_and_row_mismatches_are_fatal(tmp_path):
    with pytest.raises(SystemExit, match="outside"):
        M.cell_metrics(_cell(tmp_path / "a", [0, 1, 2, 1], np.zeros(4)), PROBE)
    with pytest.raises(SystemExit, match="logits rows"):
        M.cell_metrics(_cell(tmp_path / "b", [0, 1, 0], np.zeros(4)), PROBE)
    with pytest.raises(SystemExit, match="one class only"):
        M.cell_metrics(_cell(tmp_path / "c", [1, 1, 1], np.zeros(3)), PROBE)


def test_softmax_is_stable_at_logits_of_magnitude_1e4():
    z = np.array([[1e4, -1e4], [-1e4, 1e4], [1e4, 1e4], [-1e4, -1e4]], dtype=np.float32)
    p = M.softmax(z)
    assert np.isfinite(p).all() and np.allclose(p.sum(axis=1), 1.0)
    assert np.allclose(p[:, 1], [0.0, 1.0, 0.5, 0.5])


def test_a_label_hash_mismatch_within_one_dataset_is_fatal(tmp_path):
    """Two cells that read out different jets are not comparable. Averaging
    them would produce a number with no error, which is the dangerous case."""
    y, s = _separable()
    _cell(tmp_path, y, s, init="a")
    _cell(tmp_path, y[::-1].copy(), s[::-1].copy(), init="b")
    with pytest.raises(SystemExit, match="not paired"):
        M.main(["--root", str(tmp_path), "--out", str(tmp_path / "out")])
    assert not (tmp_path / "out" / "bench_metrics.json").exists()


def test_the_two_datasets_are_not_required_to_share_a_test_set(tmp_path):
    y, s = _separable()
    _cell(tmp_path, y, s, dataset="top")
    _cell(tmp_path, y[:2000], s[:2000], dataset="qg")
    assert M.main(["--root", str(tmp_path), "--out", str(tmp_path / "out")]) == 0
    doc = _strict(tmp_path / "out" / "bench_metrics.json")
    assert set(doc["row_alignment_sha256"]) == {"top", "qg"}
    assert doc["signal"]["qg"]["signal"] == "label_quark"
    assert doc["signal"]["top"]["signal_logit_column"] == 1


def test_a_cell_without_done_is_skipped_and_reported_and_partials_are_ignored(tmp_path):
    y, s = _separable()
    _cell(tmp_path, y, s, seed="s1")
    unfinished = _cell(tmp_path, y, s, seed="s2", done=False)
    _cell(tmp_path, y, s, seed="s3.partial.1757000000")
    assert M.main(["--root", str(tmp_path), "--out", str(tmp_path / "out")]) == 0
    doc = _strict(tmp_path / "out" / "bench_metrics.json")
    assert list(doc["cells"]["top"]["l162-s1b"]["N1000"]) == ["s1"]
    assert doc["skipped"] == [{"cell": str(unfinished), "reason": "no DONE"}]
    assert doc["summary"]["top"]["l162-s1b"]["N1000"]["auc"]["n"] == 1
    assert doc["summary"]["top"]["l162-s1b"]["N1000"]["auc"]["sd"] is None


def test_a_done_cell_without_logits_is_fatal_not_dropped(tmp_path):
    y, s = _separable()
    cell = _cell(tmp_path, y, s)
    (cell / "features" / "logits.npy").unlink()
    with pytest.raises(SystemExit, match="marked DONE"):
        M.main(["--root", str(tmp_path), "--out", str(tmp_path / "out")])


def test_the_summary_aggregates_over_seeds_and_carries_the_bound_count(tmp_path):
    """Nine head re-initialisations at N_max share the s<k> naming and ONE
    training subset; `train_subsets` is what tells them from three seeds."""
    y, _ = _separable()
    for k in range(1, 10):
        _cell(tmp_path, y, _separable(seed=k)[1], n="N1200000", seed=f"s{k}",
              subset="/data/finetune/top_sub/train_N1200000_s1.parquet")
    perfect = np.where(y == 1, 5.0, -5.0)
    for k in (1, 2, 3):
        _cell(tmp_path, y, perfect if k == 1 else _separable(seed=k)[1], seed=f"s{k}",
              subset=f"/data/finetune/top_sub/train_N1000_s{k}.parquet")
    assert M.main(["--root", str(tmp_path), "--out", str(tmp_path / "out"),
                   "--datasets", "top"]) == 0
    doc = _strict(tmp_path / "out" / "bench_metrics.json")
    cells, summ = doc["cells"]["top"]["l162-s1b"], doc["summary"]["top"]["l162-s1b"]

    big = summ["N1200000"]
    assert big["seeds"] == [f"s{k}" for k in range(1, 10)]
    assert len(big["train_subsets"]) == 1 and len(summ["N1000"]["train_subsets"]) == 3
    v = np.array([cells["N1200000"][f"s{k}"]["r30"] for k in range(1, 10)])
    assert big["r30"]["n"] == 9 and big["r30"]["n_bound"] == 0
    assert big["r30"]["mean"] == pytest.approx(v.mean())
    assert big["r30"]["sd"] == pytest.approx(v.std(ddof=1))
    assert big["r30"]["median"] == pytest.approx(np.median(v))
    assert (big["r30"]["min"], big["r30"]["max"]) == (v.min(), v.max())
    assert big["r30"]["n_bkg_pass_min"] == min(
        cells["N1200000"][f"s{k}"]["r30_n_bkg_pass"] for k in range(1, 10))

    small = summ["N1000"]
    assert small["r30"]["n_bound"] == 1 and small["r30"]["n_bkg_pass_min"] == 0
    assert small["log1m_auc"]["n_censored"] == 1
    assert not any(k.startswith("p_") or k == "p" for k in small["r30"])


def test_leg_stats_reads_one_dataset_of_the_cells_block_unchanged(tmp_path):
    y, _ = _separable()
    for k in (1, 2, 3):
        _cell(tmp_path, y, _separable(seed=k)[1], seed=f"s{k}")
    M.main(["--root", str(tmp_path), "--out", str(tmp_path / "out")])
    doc = _strict(tmp_path / "out" / "bench_metrics.json")
    ls = _load("leg_stats", "experiments/FT/leg_stats.py")
    got = ls.arm_seed_values(ls.read_cells(doc)["top"], "l162-s1b", "N1000", "r30")
    assert got == [doc["cells"]["top"]["l162-s1b"]["N1000"][f"s{k}"]["r30"] for k in (1, 2, 3)]


def test_no_cells_at_all_is_fatal(tmp_path):
    (tmp_path / "leg_top").mkdir()
    with pytest.raises(SystemExit, match="no DONE cells"):
        M.main(["--root", str(tmp_path), "--out", str(tmp_path / "out")])
