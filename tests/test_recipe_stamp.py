"""The auto-resume RECIPE stamp must be able to see a vocabulary change.

The stamp stops a run resuming a directory produced under a different recipe.
It keyed on the learning rate and the epoch budget ONLY, and that is not enough:
DECISIONS_PENDING item 24 changed the D8 control's LABEL MAP while leaving
`lr=5e-4 epochs=80` untouched, so the stamp would have matched and the relaunch
would have resumed the SUPERSEDED vocabulary's checkpoint. Both maps are K=17,
so the head shape agrees and torch loads it without error -- the control would
have silently started from the artefact it exists to replace.

These tests cover the fix AND the way the fix could itself be wrong: the snippet
emitted `RECIPE='...'` in SINGLE quotes, and bash does not expand `${MD5}`
inside those. A naive fix would have stamped the literal string `arm=${MD5}` in
every spec -- identical across vocabularies, so the guard would compare equal and
look fixed while being worse than before. The shell-execution test below is the
one that actually catches that, so it runs a real `sh`.
"""
import importlib.util
import pathlib
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _mod():
    spec = importlib.util.spec_from_file_location(
        "add_autoresume", ROOT / "scripts" / "add_autoresume.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


A = _mod()

SPEC = """apiVersion: batch/v1
kind: Job
metadata:
  name: job-fake-raunav
spec:
  backoffLimit: 0
  template:
    spec:
      containers:
      - name: train
        args:
        - |
          set -euo pipefail
          OUT=/data/results/mtx/fake
{md5}
          # seed_weaver derives four independent RNG streams
          python3 train.py --start-lr 5e-4 --num-epochs 80 --optimizer ranger
"""

MD5_BLOCK = """          CFG=configs/arms/FAKE.yaml
          MD5=$(md5sum ${CFG} | cut -d' ' -f1)"""


def _emit(tmp_path, with_md5):
    p = tmp_path / "job.yaml"
    p.write_text(SPEC.format(md5=MD5_BLOCK if with_md5 else ""))
    r = A.patch(p)
    assert not r.startswith("FAILED"), r
    return p.read_text()


def _recipe_line(text):
    return next(ln.strip() for ln in text.splitlines()
                if ln.strip().startswith("RECIPE="))


def test_a_spec_with_an_arm_config_stamps_its_fingerprint(tmp_path):
    line = _recipe_line(_emit(tmp_path, with_md5=True))
    assert "arm=${MD5}" in line
    assert line.startswith('RECIPE="'), (
        "must be DOUBLE quotes; bash does not expand ${MD5} in single quotes")


def test_a_spec_without_an_arm_config_keeps_the_two_field_stamp(tmp_path):
    line = _recipe_line(_emit(tmp_path, with_md5=False))
    assert line == "RECIPE='lr=5e-4 epochs=80'", (
        "a spec with no arm config must NOT stamp a literal empty fingerprint, "
        "which would compare equal across vocabularies")


def test_the_fingerprint_actually_expands_under_a_real_shell(tmp_path):
    """THE TEST THAT CATCHES THE QUOTING BUG. Everything else inspects strings;
    this one runs the emitted line and looks at what lands on disk."""
    line = _recipe_line(_emit(tmp_path, with_md5=True))
    out = tmp_path / "stamp"
    script = f'MD5=abc123def456\n{line}\nprintf "%s" "$RECIPE" > {out}\n'
    subprocess.run(["sh", "-c", script], check=True)
    got = out.read_text()
    assert got == "lr=5e-4 epochs=80 arm=abc123def456", got
    assert "${MD5}" not in got, (
        "the stamp contains the literal variable, so every arm would stamp the "
        "same string and the guard would compare equal across vocabularies")


def test_two_different_vocabularies_produce_different_stamps(tmp_path):
    """The whole point: item 24's failure must now be impossible."""
    line = _recipe_line(_emit(tmp_path, with_md5=True))
    stamps = []
    for digest in ("aaaa1111", "bbbb2222"):
        out = tmp_path / f"s_{digest}"
        subprocess.run(
            ["sh", "-c", f'MD5={digest}\n{line}\nprintf "%s" "$RECIPE" > {out}\n'],
            check=True)
        stamps.append(out.read_text())
    assert stamps[0] != stamps[1], (
        "two vocabularies stamped identically; the guard cannot see a label "
        "change and item 24's silent resume is possible again")


def test_the_guard_refuses_a_mismatched_stamp_end_to_end(tmp_path):
    """Exercise the emitted comparison, not just the stamp construction."""
    text = _emit(tmp_path, with_md5=True)
    body = "\n".join(ln for ln in text.splitlines()
                     if not ln.strip().startswith("- |"))
    start = body.index("          RECIPE=")
    end = body.index("printf '%s'", start)
    guard = "\n".join(ln[10:] if ln.startswith("          ") else ln
                      for ln in body[start:end].splitlines())

    out = tmp_path / "run"
    out.mkdir()
    (out / "net_epoch-0_state.pt").write_bytes(b"x")
    (out / "net_epoch-0_optimizer.pt").write_bytes(b"x")
    (out / "RECIPE").write_text("lr=5e-4 epochs=80 arm=OLDVOCAB")

    r = subprocess.run(["sh", "-c", f'OUT={out}\nMD5=NEWVOCAB\n{guard}'],
                       capture_output=True, text=True)
    assert r.returncode == 1, f"guard did not refuse:\n{r.stdout}\n{r.stderr}"
    assert "DIFFERENT recipe" in r.stdout
    assert "OLDVOCAB" in r.stdout and "NEWVOCAB" in r.stdout


def test_the_control_spec_already_carries_the_fingerprint():
    """job-mtx-rand-d1-s1b was hand-patched when item 24 was found; it must stay
    consistent with what the generator now emits for every arm."""
    p = ROOT / "experiments/MTX/k8s/job-mtx-rand-d1-s1b-raunav.yaml"
    line = _recipe_line(p.read_text())
    assert "arm=${MD5}" in line and line.startswith('RECIPE="')


def test_a_rate_mentioned_in_a_comment_does_not_break_regeneration():
    """These specs discuss learning rates at length in their headers. A bare
    scan for --start-lr picked those up and `patch()` refused to run on a real
    spec, reporting four distinct rates including the words 'but' and 'this'.
    Detection is comment-stripped; replacement still sees the full text."""
    import re
    import tempfile
    src = (ROOT / "experiments/MTX/k8s/job-mtx-l162-s2-raunav.yaml").read_text()
    assert src.count("--start-lr") > 1, (
        "fixture no longer exercises the bug: this spec must mention the flag "
        "in a comment as well as on the command line")

    start = src.index("          # AUTO-RESUME + RECIPE GUARD")
    end = src.index("          # seed_weaver derives four independent RNG streams")
    stripped = src[:start] + src[end:]
    stripped = stripped.replace("--num-epochs 80 ${RESUME} --optimizer ranger",
                                "--num-epochs 80 --optimizer ranger")
    stripped = re.sub(r"^  backoffLimit: \d+$", "  backoffLimit: 0",
                      stripped, flags=re.M)

    p = pathlib.Path(tempfile.mkdtemp()) / "j.yaml"
    p.write_text(stripped)
    r = A.patch(p)
    assert not r.startswith("FAILED"), r
    line = _recipe_line(p.read_text())
    assert line == 'RECIPE="lr=5e-4 epochs=80 arm=${MD5}"', line


def test_a_spec_assigning_md5_after_the_anchor_is_refused(tmp_path):
    """Shell order matters: the stamp would expand to an empty fingerprint,
    which compares equal across vocabularies."""
    bad = SPEC.format(md5="").replace(
        "          # seed_weaver derives four independent RNG streams",
        "          # seed_weaver derives four independent RNG streams\n" + MD5_BLOCK)
    p = tmp_path / "bad.yaml"
    p.write_text(bad)
    r = A.patch(p)
    assert r.startswith("FAILED") and "AFTER the auto-resume anchor" in r
