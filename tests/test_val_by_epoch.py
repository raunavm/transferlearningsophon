"""The S8 log parser: per-epoch validation accuracy and the validation reader's position."""
import importlib.util
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("val_by_epoch", ROOT / "experiments/MTX/val_by_epoch.py")
V = importlib.util.module_from_spec(spec)
spec.loader.exec_module(V)


def _restart(worker, files):
    return ([f"[t] INFO: Restarted DataIter {worker}, load_range=(0.0, 1.0), file_list:"]
            + json.dumps({"_": files}, indent=2).splitlines())


def _epoch(e, metric, restart=None):
    out = [f"[t] INFO: Epoch #{e} training", f"[t] INFO: Epoch #{e} validating"]
    out += restart or []
    return out + [f"[t] INFO: \x1b[1mEpoch #{e}: Current validation metric: {metric} (best: 0.9)\x1b[0m"]


def test_reader_position_resets_at_a_resume_and_the_last_value_of_an_epoch_wins(tmp_path):
    a = _restart("val_worker0", ["QCD_1.parquet", "QCD_2.parquet"]) + _restart("val_worker1", ["X_1.parquet"])
    lines = (_epoch(0, 0.40, a) + _epoch(1, 0.45) + _epoch(2, 0.47)
             # crash during epoch 3, resume from epoch 2's checkpoint: epoch 2 runs again
             + ["[t] INFO: Resume training from epoch 2"] + _epoch(2, 0.46, a) + _epoch(3, 0.50))
    log = tmp_path / "train.log"
    log.write_text("\n".join(lines) + "\n")
    r = V.parse(log)
    assert r["metric"] == {0: 0.40, 1: 0.45, 2: 0.46, 3: 0.50}
    assert r["resumes"] == [2]
    sha = r["val_state"][0][0]
    assert sha is not None
    # epoch 2 was re-validated by a restarted reader: position 1, not 3
    assert r["val_state"] == {0: [sha, 1], 1: [sha, 2], 2: [sha, 1], 3: [sha, 2]}


def test_a_different_file_list_gives_a_different_reader_state(tmp_path):
    for name, files in (("a", ["QCD_1.parquet"]), ("b", ["QCD_2.parquet"])):
        (tmp_path / name).write_text("\n".join(_epoch(0, 0.4, _restart("val_worker0", files))) + "\n")
    assert V.parse(tmp_path / "a")["val_state"][0] != V.parse(tmp_path / "b")["val_state"][0]


def test_train_worker_file_lists_are_not_mistaken_for_the_validation_reader(tmp_path):
    lines = (_restart("train_worker0", ["T.parquet"])
             + _epoch(0, 0.4, _restart("val_worker0", ["V.parquet"])))
    (tmp_path / "log").write_text("\n".join(lines) + "\n")
    lone = tmp_path / "lone"
    lone.write_text("\n".join(_epoch(0, 0.4, _restart("val_worker0", ["V.parquet"]))) + "\n")
    assert V.parse(tmp_path / "log")["val_state"] == V.parse(lone)["val_state"]
