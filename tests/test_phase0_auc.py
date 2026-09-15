"""Phase 0 part A: the 14 LR-sweep cells restated in D7's metrics.

THE FAILURE THIS FILE GUARDS AGAINST IS A SILENT SIGNAL SWAP. The discriminant
is `score_label_Top` and the truth is `label_Top`, both named by
configs/finetune/TopReference.yaml's `value: [label_QCD, label_Top]`. Reading
"the second score branch" positionally would keep working if that order were
reversed and would report 1 - AUC with nothing raised -- an exactly inverted
sweep table in which every value still sits in [0.5, 1] and looks plausible.

The metric arithmetic is exercised through `row_for`, which takes arrays, so it
runs without a ROOT file. That is not a testability flourish: the installed
uproot (5.1.2) CANNOT round-trip one under numpy 2.3.5 -- `recreate` raises on
np.VisibleDeprecationWarning, and shimming that attribute makes it write a file
that then fails to read back with an OverflowError. A test suite that depended
on writing .root files here would simply not run on this machine.
"""
import importlib.util
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _mod():
    spec = importlib.util.spec_from_file_location(
        "phase0_auc", ROOT / "experiments" / "FT" / "phase0_auc.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


A = _mod()


def _separable(n=2000, sep=1.5, seed=0):
    rng = np.random.default_rng(seed)
    y = np.repeat([0, 1], n // 2)
    s = rng.normal(0.0, 1.0, n) + sep * y
    return y, 1.0 / (1.0 + np.exp(-s))


# ------------------------------------------------------------- the cell parser

@pytest.mark.parametrize("name,arm,lr", [
    ("l162-s1b_lr3e-05", "l162-s1b", 3e-5),
    ("r16q1-s2_lr0.01", "r16q1-s2", 1e-2),
    ("l162-s1b_lr0.0003", "l162-s1b", 3e-4),
])
def test_the_cell_name_yields_the_arm_and_the_rate(name, arm, lr):
    m = A.CELL.match(name)
    assert m and m["arm"] == arm and float(m["lr"]) == pytest.approx(lr)


def test_a_directory_that_is_not_a_cell_is_not_matched():
    assert A.CELL.match("summary") is None
    assert A.CELL.match("net_epoch-3_state.pt") is None


# --------------------------------------------------- refusal by name, not index

def test_both_required_branches_present_is_accepted():
    A.check_branches({"label_QCD", "label_Top",
                      "score_label_QCD", "score_label_Top", "jet_pt"})


def test_a_missing_score_branch_is_refused_by_name():
    """THE LOAD-BEARING TEST. A positional fallback would read QCD as the signal
    and report 1 - AUC with no error raised anywhere."""
    with pytest.raises(SystemExit, match="score_label_Top"):
        A.check_branches({"label_Top", "score_0", "score_1"})


def test_a_missing_truth_branch_is_refused():
    with pytest.raises(SystemExit, match="label_Top"):
        A.check_branches({"score_label_Top", "score_label_QCD"})


def test_the_refusal_names_what_it_did_find():
    """A bare 'missing branch' forces a cluster round trip to learn the names."""
    with pytest.raises(SystemExit, match="score_0"):
        A.check_branches({"score_0", "score_1"})


# ------------------------------------------------------------- the metric row

def test_a_row_carries_the_three_d7_quantities():
    y, s = _separable()
    r = A.row_for(y, s, "lrprobe", "l162-s1b", 3e-5)
    assert 0.5 < r["auc"] <= 1.0
    assert r["log1m_auc"] < 0
    assert r["rejection_at_0.50"] > 1.0
    assert r["n_sig"] == 1000 and r["n_bkg"] == 1000


def test_a_single_class_cell_is_refused_and_names_the_cell():
    """roc_auc_score raises on one class, but only after the row looks real.
    Refusing early names the cell, which is what makes it fixable."""
    with pytest.raises(SystemExit, match=r"l162-s1b_lr0\.001 has one class"):
        A.row_for(np.ones(500, int), np.linspace(0, 1, 500),
                  "lrprobe", "l162-s1b", 0.001)


def test_a_better_separated_cell_scores_better_on_all_three():
    """Direction check. If signal and background were swapped this inverts."""
    lo = A.row_for(*_separable(sep=0.5, seed=7), "lrprobe", "a", 1e-3)
    hi = A.row_for(*_separable(sep=3.0, seed=7), "lrprobe", "a", 1e-2)
    assert hi["auc"] > lo["auc"]
    assert hi["rejection_at_0.50"] > lo["rejection_at_0.50"]
    assert hi["log1m_auc"] < lo["log1m_auc"]


def test_an_inverted_score_is_visibly_worse_than_chance():
    """What a silent signal swap would actually look like, pinned so the
    direction assertions above cannot be read as arbitrary."""
    y, s = _separable(sep=2.0)
    assert A.row_for(y, 1.0 - s, "lrprobe", "a", 1e-3)["auc"] < 0.5


# -------------------------------------------------------------------- the grid

def test_the_grid_walks_cells_and_keys_them_by_arm_and_rate(tmp_path, monkeypatch):
    cells = [("lrprobe", "l162-s1b", "3e-05"), ("lrprobe", "r16q1-s2", "3e-05"),
             ("lrprobe3", "l162-s1b", "0.03")]
    for probe, arm, lr in cells:
        d = tmp_path / probe / f"{arm}_lr{lr}"
        d.mkdir(parents=True)
        (d / "pred.root").write_bytes(b"")
    monkeypatch.setattr(A, "read_cell", lambda p: _separable(seed=len(str(p))))
    rows = A.grid(tmp_path)
    assert {(r["arm"], r["lr"]) for r in rows} == {
        ("l162-s1b", 3e-5), ("r16q1-s2", 3e-5), ("l162-s1b", 3e-2)}
    assert {r["probe"] for r in rows} == {"lrprobe", "lrprobe3"}


def test_the_grid_ignores_a_directory_that_is_not_a_cell(tmp_path, monkeypatch):
    for sub in ("l162-s1b_lr0.001", "summary"):
        d = tmp_path / "lrprobe" / sub
        d.mkdir(parents=True)
        (d / "pred.root").write_bytes(b"")
    monkeypatch.setattr(A, "read_cell", lambda p: _separable())
    assert [r["arm"] for r in A.grid(tmp_path)] == ["l162-s1b"]


# ------------------------------------------------------------------ provenance

def test_the_metrics_are_imported_from_probe_not_reimplemented():
    """scripts/rand_control_stats.py exists because a metric was hand-formed
    once. Two copies of log1m_auc would drift and both would look right."""
    src = (ROOT / "experiments" / "FT" / "phase0_auc.py").read_text()
    assert "_probe()" in src and "p.log1m_auc(" in src and "p.rejection_at(" in src
    for banned in ("def log1m_auc", "def rejection_at", "roc_auc_score("):
        assert banned not in src, f"phase0_auc.py reimplements {banned!r}"


def test_probe_still_exports_what_this_depends_on():
    """If probe.py renames either function this fails here rather than on the
    cluster after a 14-cell GPU pass."""
    p = A._probe()
    assert callable(p.log1m_auc) and callable(p.rejection_at)
    assert p.EPS_S == 0.5


# ------------------------------------------------- the overlap anchors, 18 -> 14

def _grid_like_the_real_one(tmp_path):
    """The sweep's actual shape: 3 probes x 2 arms x 3 rates = 18 directories,
    14 distinct (arm, rate) points, because each probe repeats its
    predecessor's top rate as an overlap anchor."""
    plan = {"lrprobe": ["3e-5", "1e-4", "3e-4"],
            "lrprobe2": ["3e-4", "1e-3", "3e-3"],
            "lrprobe3": ["3e-3", "1e-2", "3e-2"]}
    for probe, rates in plan.items():
        for arm in ("l162-s1b", "r16q1-s2"):
            for lr in rates:
                d = tmp_path / probe / f"{arm}_lr{lr}"
                d.mkdir(parents=True)
                (d / "pred.root").write_bytes(b"")
    return plan


def test_the_real_grid_is_eighteen_directories_and_fourteen_points(tmp_path, monkeypatch):
    """The number the job spec's guard checks. 14 alone would pass a grid that
    had lost its anchors; 18 alone would pass one that gained a rate."""
    _grid_like_the_real_one(tmp_path)
    monkeypatch.setattr(A, "read_cell", lambda p: _separable(seed=len(str(p))))
    rows = A.grid(tmp_path)
    assert len(rows) == 18
    assert len({(r["arm"], r["lr"]) for r in rows}) == 14


def test_a_repeated_point_keeps_both_rows_and_is_not_averaged(tmp_path, monkeypatch):
    """Averaging the anchors away would destroy the only measurement of
    run-to-run variance in the table."""
    _grid_like_the_real_one(tmp_path)
    monkeypatch.setattr(A, "read_cell", lambda p: _separable(seed=len(str(p))))
    rows = A.grid(tmp_path)
    for arm in ("l162-s1b", "r16q1-s2"):
        for lr in (3e-4, 3e-3):
            got = [r for r in rows if r["arm"] == arm and r["lr"] == lr]
            assert len(got) == 2, f"{arm} lr={lr} should appear in two probes"
            assert len({r["probe"] for r in got}) == 2


def test_main_prints_the_anchor_section_without_crashing(tmp_path, monkeypatch, capsys):
    """Covers the report's format strings end to end. Every number printed there
    goes through a width/precision spec, and one bad spec would take out the
    whole run AFTER the 18 GPU cells had been scored."""
    _grid_like_the_real_one(tmp_path)
    monkeypatch.setattr(A, "read_cell", lambda p: _separable(seed=len(str(p))))
    out = tmp_path / "phase0_auc.json"
    assert A.main(["--grid", str(tmp_path), "--out", str(out)]) == 0
    txt = capsys.readouterr().out
    assert "overlap anchors" in txt
    assert "per-arm spread" in txt
    assert "WARNING: no overlap anchor" not in txt
    import json
    assert len(json.loads(out.read_text())) == 18


def test_main_warns_when_no_anchor_is_present(tmp_path, monkeypatch, capsys):
    """A grid with no repeated point cannot support any significance claim, and
    must say so rather than printing a clean-looking table."""
    for lr in ("3e-5", "1e-4"):
        d = tmp_path / "lrprobe" / f"l162-s1b_lr{lr}"
        d.mkdir(parents=True)
        (d / "pred.root").write_bytes(b"")
    monkeypatch.setattr(A, "read_cell", lambda p: _separable(seed=len(str(p))))
    A.main(["--grid", str(tmp_path)])
    assert "WARNING: no overlap anchor" in capsys.readouterr().out


def test_an_empty_grid_is_refused_rather_than_written_as_an_empty_table(tmp_path):
    with pytest.raises(SystemExit, match="no .*pred.root"):
        A.main(["--grid", str(tmp_path)])


# ------------------------------------- what a zero anchor spread does NOT license

def test_a_zero_spread_is_reported_as_determinism_not_as_a_variance_bound(
        tmp_path, monkeypatch, capsys):
    """THE DEFECT THIS PINS WAS IN MY OWN REPORT, and the real run triggered it.
    The message read "any arm gap smaller than the spread is unreadable"; the
    spread came back EXACTLY 0.00000, which licenses every gap in the table.
    Backwards: a zero spread says the two probe jobs rebuilt the same checkpoint
    bit for bit. Every cell ran --seed 1, so seed variance is unmeasured, and
    reproducibility says nothing about the test set's own statistical error."""
    _grid_like_the_real_one(tmp_path)
    # identical scores for identical (arm, lr) -> spread exactly zero, as in production
    monkeypatch.setattr(A, "read_cell",
                        lambda p: _separable(seed=hash(p.parent.name) % 997))
    A.main(["--grid", str(tmp_path)])
    txt = capsys.readouterr().out
    assert "Spread is EXACTLY zero" in txt
    assert "determinism at fixed" in txt and "seed variance is unmeasured" in txt
    assert "unreadable and must not be quoted" not in txt, (
        "the variance-bound wording must not appear when the spread is zero")


def test_a_real_spread_still_gets_the_variance_wording(tmp_path, monkeypatch, capsys):
    _grid_like_the_real_one(tmp_path)
    monkeypatch.setattr(A, "read_cell", lambda p: _separable(seed=len(str(p))))
    A.main(["--grid", str(tmp_path)])
    txt = capsys.readouterr().out
    assert "Spread is EXACTLY zero" not in txt
    assert "unreadable and must not be quoted" in txt


# ------------------------------------------------ the arm gap and its error bar

def test_the_arm_gap_is_reported_against_the_poisson_band(tmp_path, monkeypatch, capsys):
    """Rejection is 1/eps_B and eps_B comes from a COUNT, so a cell with ~1,000
    surviving background jets carries a ~3% band -- the size of the arm gaps
    themselves at the top of the grid. Quoting the gap bare is how a 0.8-sigma
    difference becomes 'the ordering flips'."""
    _grid_like_the_real_one(tmp_path)
    monkeypatch.setattr(A, "read_cell", lambda p: _separable(seed=len(str(p))))
    A.main(["--grid", str(tmp_path)])
    txt = capsys.readouterr().out
    assert "arm gap in rejection" in txt and "n_sigma" in txt
    assert "upper bounds on" in txt, (
        "the band is background-only; signal error and seed variance widen it, "
        "so the printed n_sigma must be flagged as an upper bound")


def test_the_gap_table_is_skipped_when_a_rate_has_only_one_arm(tmp_path, monkeypatch, capsys):
    for lr in ("3e-5", "1e-4"):
        d = tmp_path / "lrprobe" / f"l162-s1b_lr{lr}"
        d.mkdir(parents=True)
        (d / "pred.root").write_bytes(b"")
    monkeypatch.setattr(A, "read_cell", lambda p: _separable(seed=len(str(p))))
    A.main(["--grid", str(tmp_path)])
    assert "arm gap in rejection" not in capsys.readouterr().out
