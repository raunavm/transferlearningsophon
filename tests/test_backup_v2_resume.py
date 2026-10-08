"""experiments/MTX/backup_v2_resume.py: verified copies of what a v2 run cannot regenerate."""
import importlib.util
import json
import pathlib

import pytest

torch = pytest.importorskip("torch")
ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("backup_v2_resume", ROOT / "experiments/MTX/backup_v2_resume.py")
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)


def _model(e):
    return {"w": torch.full((3,), float(e)), "b": torch.tensor([e + 0.5])}


def _epoch(run, e, *, resume=True):
    torch.save(_model(e), run / f"net_epoch-{e}_state.pt")
    if resume:
        torch.save({"epoch": e, "model": _model(e), "optimizer": {}, "scheduler": {}, "scaler": {},
                    "trimmer_counters": {}, "best": {"epoch": e}}, run / f"net_epoch-{e}_resume.pt")
        for p in run.glob("net_epoch-*_resume.pt"):           # the run keeps only its newest
            if p.name != f"net_epoch-{e}_resume.pt":
                p.unlink()
    (run / "best_epoch.json").write_text(json.dumps({"epoch": e}))


def _run(tmp_path):
    run = tmp_path / "mtx-l162-s1"
    run.mkdir()
    (run / "recipe.json").write_text(json.dumps({"seed": 1}))
    return run


def _top(run):
    return {p.name: p.read_bytes() for p in run.iterdir() if p.is_file()}


def test_the_newest_resume_pair_kept_states_and_recipe_are_copied_verified_and_the_run_untouched(tmp_path):
    run = _run(tmp_path)
    for e in (0, 1, 2):
        _epoch(run, e)
    before = _top(run)
    assert B.backup_run(run) == []
    assert _top(run) == before                                       # nothing in the run changes
    b = run / "backup"
    assert (b / "resume/net_epoch-2_resume.pt").read_bytes() == (run / "net_epoch-2_resume.pt").read_bytes()
    assert (b / "resume/net_epoch-2_state.pt").read_bytes() == (run / "net_epoch-2_state.pt").read_bytes()
    assert (b / "states/net_epoch-0_state.pt").exists() and (b / "states/net_epoch-2_state.pt").exists()
    assert not (b / "states/net_epoch-1_state.pt").exists()          # retention does not keep it for good
    assert json.loads((b / "recipe.json").read_text()) == {"seed": 1}
    man = json.loads((b / "manifest.json").read_text())["copies"]
    for rel, rec in man.items():
        assert rec["sha256"] == B.sha256(b / rel)
    assert not list(b.rglob("*.tmp"))
    # the training script and the job script look only at the top level
    assert B.latest_complete_epoch(run) == 2 and len(list(run.glob("net_epoch-*_resume.pt"))) == 1


def test_a_second_pass_copies_nothing_new_and_only_two_resume_epochs_are_kept(tmp_path):
    run = _run(tmp_path)
    _epoch(run, 0)
    B.backup_run(run)
    stamp = json.loads((run / "backup/manifest.json").read_text())
    assert B.backup_run(run) == [] and json.loads((run / "backup/manifest.json").read_text()) == stamp
    for e in (1, 2, 3):
        _epoch(run, e)
        B.backup_run(run)
    kept = sorted(p.name for p in (run / "backup/resume").iterdir())
    assert kept == ["net_epoch-2_resume.pt", "net_epoch-2_state.pt", "net_epoch-3_resume.pt",
                    "net_epoch-3_state.pt"]
    man = json.loads((run / "backup/manifest.json").read_text())["copies"]
    assert {k for k in man if k.startswith("resume/")} == {f"resume/{n}" for n in kept}
    assert (run / "backup/states/net_epoch-0_state.pt").exists()      # kept states stay


def test_a_zero_filled_or_inconsistent_resume_file_is_reported_and_not_copied(tmp_path):
    run = _run(tmp_path)
    _epoch(run, 0)
    B.backup_run(run)
    _epoch(run, 1)
    size = (run / "net_epoch-1_resume.pt").stat().st_size
    (run / "net_epoch-1_resume.pt").write_bytes(b"\0" * size)        # what a full pool left on 2026-10-07
    problems = B.backup_run(run)
    assert len(problems) == 1 and "net_epoch-1_resume.pt does not load" in problems[0]
    assert not (run / "backup/resume/net_epoch-1_resume.pt").exists()
    assert (run / "backup/resume/net_epoch-0_resume.pt").exists()     # the last good one survives
    _epoch(run, 2)
    torch.save(_model(99), run / "net_epoch-2_state.pt")              # a state that is not the resume's model
    assert any("not the state file" in p for p in B.backup_run(run))
    assert not (run / "backup/resume/net_epoch-2_resume.pt").exists()


def test_a_zero_filled_recipe_is_reported_and_a_state_beyond_the_newest_epoch_waits(tmp_path):
    run = _run(tmp_path)
    (run / "recipe.json").write_bytes(b"\0" * 64)
    _epoch(run, 0)
    torch.save(_model(70), run / "net_epoch-70_state.pt")             # a later attempt may rewrite it
    problems = B.backup_run(run)
    assert any("recipe.json does not parse" in p for p in problems)
    assert not (run / "backup/recipe.json").exists()
    assert not (run / "backup/states/net_epoch-70_state.pt").exists()


def test_main_reports_problems_by_exit_code(tmp_path, capsys):
    run = _run(tmp_path)
    _epoch(run, 0)
    assert B.main(["--root", str(tmp_path)]) == 0
    (run / "net_epoch-0_resume.pt").write_bytes(b"\0" * 16)
    _epoch(run, 1, resume=False)
    torch.save({"epoch": 1}, run / "net_epoch-1_resume.pt")           # loads, but lacks the model
    assert B.main(["--root", str(tmp_path)]) == 1
    assert "PROBLEM mtx-l162-s1/net_epoch-1_resume.pt lacks" in capsys.readouterr().out
