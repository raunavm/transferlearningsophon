"""experiments/FT/cell_resume.py and ft_weaver.py, the pieces a resumed
fine-tuning cell rests on (retry logic of 2026-10-01). The shell behaviour --
resume after an eviction, halts -- is in tests/test_ft_retry_policy.py; this
pins the rules themselves:

  * best: weaver's own rule (first strict maximum from 0) on the exact metric,
    every epoch exactly once, the copy made only for a resumed run;
  * prepare: the resume epoch is the last one logged as finished whose files
    are whole, the cell is cut back to it, a failed attempt is moved aside;
  * ft_weaver: the stock evaluate's signature survives the wrapper (seed_weaver
    --lean-val-metrics reads it) and the training-time metric is logged exactly.
"""
import importlib.util
import inspect
import json
import logging
import pathlib
import zipfile

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


CR = _load("cell_resume", "experiments/FT/cell_resume.py")


def _zip(path, text):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("w", text)


def _cell(tmp_path, metrics, finished=None, resumed=False, best=None):
    """A cell after weaver: epoch files, a train.log with both metric lines."""
    c = tmp_path / "cell"
    c.mkdir()
    log = ["[t] INFO: args:\n"]
    for e, m in enumerate(metrics):
        if resumed and e == 2:
            log.append("[t] INFO: Resume training from epoch 1\n")
        log.append(f"[t] INFO: Epoch #{e} training\n")
        if finished is not None and e >= finished:
            break
        _zip(c / f"net_epoch-{e}_state.pt", f"state {e}")
        _zip(c / f"net_epoch-{e}_optimizer.pt", f"opt {e}")
        log.append(f"[t] INFO: Epoch #{e}: exact validation metric {m!r}\n")
        log.append(f"[t] INFO: \x1b[1mEpoch #{e}: Current validation metric: {m:.5f} (best: 0)\x1b[0m\n")
    (c / "train.log").write_text("".join(log))
    if best is not None:
        (c / "net_best_epoch_state.pt").write_bytes((c / f"net_epoch-{best}_state.pt").read_bytes())
    return c


def _best_state(c):
    with zipfile.ZipFile(c / "net_best_epoch_state.pt") as z:
        return z.read("w").decode()


def test_best_is_weavers_rule_on_the_exact_metric(tmp_path):
    # equal at five decimals, apart at full precision: the later epoch is better
    m = [0.5, 0.7712, 0.77121, 0.771210001, 0.6]
    c = _cell(tmp_path, m, best=3)
    rec = CR.best(c, 5)
    assert rec["epoch"] == 3 and not rec["restored"] and _best_state(c) == "state 3"
    assert json.loads((c / "best_epoch.json").read_text())["metrics"] == m


def test_best_keeps_the_first_of_tied_epochs(tmp_path):
    c = _cell(tmp_path, [0.5, 0.8, 0.8, 0.7], best=1)
    assert CR.best(c, 4)["epoch"] == 1


def test_best_restores_the_runs_best_after_a_resume(tmp_path):
    # weaver restarted at 0 after epoch 1 and kept epoch 2; epoch 1 is the run's best
    c = _cell(tmp_path, [0.5, 0.9, 0.6, 0.7], resumed=True, best=3)
    rec = CR.best(c, 4)
    assert rec["epoch"] == 1 and rec["resumed"] and rec["restored"] and _best_state(c) == "state 1"


def test_best_refuses_a_disagreement_in_a_run_that_never_resumed(tmp_path):
    c = _cell(tmp_path, [0.5, 0.9, 0.6], best=2)
    with pytest.raises(SystemExit, match="never resumed"):
        CR.best(c, 3)


def test_best_wants_every_epoch_exactly_once(tmp_path):
    c = _cell(tmp_path, [0.5, 0.9, 0.6], best=1)
    with pytest.raises(SystemExit, match="not 0..3"):
        CR.best(c, 4)
    with open(c / "train.log", "a") as f:
        f.write("[t] INFO: Epoch #1: exact validation metric 0.9\n")
    with pytest.raises(SystemExit, match="twice"):
        CR.best(c, 3)


def test_prepare_cuts_the_cell_back_to_its_last_whole_epoch(tmp_path):
    c = _cell(tmp_path, [0.5, 0.6, 0.7, 0.8, 0.9], finished=4)      # killed in epoch 4
    (c / "stdout.log").write_text("out\n")
    (c / "features").mkdir()
    (c / "net_epoch-3_optimizer.pt").write_bytes(b"PK\x03\x04 cut short")
    (c / "ATTEMPTS").write_text("t pod=p node=n from_epoch=-1\n")
    assert CR.prepare(c, 3) == 2
    log = (c / "train.log").read_text()
    assert log.rstrip().endswith("Epoch #2: Current validation metric: 0.70000 (best: 0)\x1b[0m")
    assert "Epoch #3" in (c / "train.log.cut.1").read_text()
    assert (c / "stdout.log.1").exists() and not (c / "stdout.log").exists()
    assert not (c / "features").exists() and not list(c.glob("net_epoch-3_*"))
    assert (c / "net_epoch-2_state.pt").exists()


def test_prepare_moves_a_failed_attempt_aside(tmp_path):
    c = _cell(tmp_path, [0.5, 0.6])
    (c / "ATTEMPT_FAILED").write_text("rc=1\n")
    assert CR.prepare(c, 3) == -1
    assert not c.exists() and len(list(tmp_path.glob("cell.partial.*"))) == 1


def test_stalled_counts_only_the_latest_attempts_without_progress():
    assert CR.stalled([], 5) == 0
    assert CR.stalled([-1, 4, 11], 19) == 0
    assert CR.stalled([-1, 2, 2], 2) == 2
    assert CR.stalled([-1, 2, 2, 2], 2) == 3
    assert CR.stalled([-1, -1, -1], -1) == 3


def test_ft_weaver_keeps_the_stock_signature_and_logs_the_exact_metric(monkeypatch, caplog):
    tools = pytest.importorskip("weaver.utils.nn.tools")
    FW = _load("ft_weaver", "experiments/FT/ft_weaver.py")
    assert "eval_metrics" in inspect.signature(FW._evaluate).parameters
    monkeypatch.setattr(FW, "_stock", lambda model, loader, dev, epoch, *a, **kw: 2071 / 2560
                        if kw.get("for_training", True) else (0.5, None, None, None))
    logger = logging.getLogger("weaver")
    monkeypatch.setattr(logger, "propagate", True)
    with caplog.at_level(logging.INFO, logger="weaver"):
        assert FW._evaluate(None, None, "cpu", 7) == 2071 / 2560
        FW._evaluate(None, None, "cpu", 8, for_training=False)
    lines = [r.getMessage() for r in caplog.records if "exact validation metric" in r.getMessage()]
    assert lines == [f"Epoch #7: exact validation metric {2071 / 2560!r}"]
    assert CR.EXACT.search(lines[0]).group(2) == repr(2071 / 2560)
    assert tools.evaluate_classification is not FW._evaluate     # main() installs it, import does not
