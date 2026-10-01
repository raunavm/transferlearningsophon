"""Every spec that mounts /data read-write checks the volume's fill before its first
write (CLAUDE.md: refuse writes when the volume is past 85 %).

v1err-batch-b wrote 184 MiB of mass residuals with no check at all; 151 specs in
experiments/EVAL/k8s ran that way. They are history and are listed, frozen, in
experiments/EVAL/k8s/ran_without_storage_guard.txt; every other spec must pass,
and the builders put the guard in (scripts/build_extract_jobs.py storage_guarded).
"""
import importlib.util
import pathlib
import re
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from test_spec_pins import _launched_stems  # noqa: E402
K8S = ROOT / "experiments" / "EVAL" / "k8s"
GUARD = re.compile(r"df --output=pcent /data\b")
WRITE = re.compile(r"mkdir|\btee\b|>>|(?<![<>0-9&])>(?![&=])|python3 +(?:experiments|scripts)/"
                   r"|^weaver |\bcp\b|\bmv\b|\brm\b")
# Never launched, and never launchable as they stand: weaver --predict of weaver's
# best-epoch file on the 80-epoch matrix, which item 18's checkpoint rule replaced and
# scripts/build_eval_jobs.py now refuses to emit (it requires --ckpt-epoch for mtx).
OBSOLETE = {"job-eval-mtx-l162-s1-raunav.yaml", "job-eval-mtx-l162-s1b-raunav.yaml",
            "job-eval-mtx-r16q1-s4-raunav.yaml", "job-eval-mtx-r16q1-s5-raunav.yaml"}


def _bx():
    s = importlib.util.spec_from_file_location("build_extract_jobs", ROOT / "scripts" / "build_extract_jobs.py")
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def guarded_before_first_write(text: str) -> bool | None:
    """None if the spec does not mount /data read-write, else whether a storage
    check precedes the first command that can write."""
    d = yaml.safe_load(text)
    if d.get("kind") != "Job":
        return None
    c = d["spec"]["template"]["spec"]["containers"][0]
    if not any(m["mountPath"] == "/data" and not m.get("readOnly") for m in c.get("volumeMounts", [])):
        return None
    lines = [l.strip() for l in "\n".join(c.get("command", []) + c.get("args", [])).splitlines()]
    lines = [l for l in lines if l and not l.startswith("#")]
    g = next((i for i, l in enumerate(lines) if GUARD.search(l)), None)
    w = next((i for i, l in enumerate(lines) if WRITE.search(l) and not GUARD.search(l)
              and not l.startswith(("git clone", "pip install"))), None)
    return g is not None and (w is None or g < w)


def _launched(name: str) -> bool:
    """A run record names the spec: test_spec_pins' rule (run_id or manifest_path,
    a run_id versioning the spec name as <stem>-v.. or <stem>-s..)."""
    stem = name.removeprefix("job-").removesuffix("-raunav.yaml")
    return any(r == stem or r.startswith(stem + "-v") or r.startswith(stem + "-s")
               for r in _launched_stems())


def test_every_spec_that_writes_to_data_checks_space_first():
    frozen = _bx().RAN_WITHOUT_STORAGE_GUARD
    bad = [s.name for s in sorted(K8S.glob("*.yaml"))
           if guarded_before_first_write(s.read_text()) is False
           and s.name not in frozen and s.name not in OBSOLETE]
    assert not bad, f"no /data space check before the first write: {bad}"


def test_the_frozen_list_is_history_and_does_not_grow():
    frozen = _bx().RAN_WITHOUT_STORAGE_GUARD
    assert len(frozen) <= 151, "a spec was added to the frozen list: it shipped unguarded"
    for name in frozen:
        assert (K8S / name).exists(), f"{name} is frozen but does not exist"
        assert _launched(name), f"{name} is frozen but has no run record: guard it instead"
    for name in OBSOLETE:
        text = (K8S / name).read_text()
        assert not _launched(name) and "net_best_epoch_state.pt" in text and "/data/results/mtx/" in text


def test_the_rule_sees_a_write_before_the_guard_and_the_builders_insert_it():
    bx = _bx()
    late = ("apiVersion: batch/v1\nkind: Job\nmetadata: {name: x-raunav}\nspec:\n  template:\n"
            "    spec:\n      containers:\n      - name: main\n        command: [\"/bin/bash\", \"-c\"]\n"
            "        args:\n        - |\n          set -euo pipefail\n          git clone x /workspace/transferlearningsophon\n"
            "          cd /workspace/transferlearningsophon\n          mkdir -p /data/results/x\n"
            "        volumeMounts:\n        - { name: data, mountPath: /data }\n")
    assert guarded_before_first_write(late) is False
    assert guarded_before_first_write(bx.storage_guarded("job-x-raunav.yaml", late)) is True
    assert guarded_before_first_write(late.replace("mountPath: /data }", "mountPath: /data, readOnly: true }")) is None
    # a spec that already ran unguarded keeps its text
    name = sorted(bx.RAN_WITHOUT_STORAGE_GUARD)[0]
    text = (K8S / name).read_text()
    assert bx.storage_guarded(name, text) == text
