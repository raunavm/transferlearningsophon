"""scripts/build_indexed_runner.py: one Indexed Job whose index i runs the i-th spec's script."""
from __future__ import annotations

import importlib.util
import pathlib
import subprocess

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_indexed_runner", ROOT / "scripts/build_indexed_runner.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
HEAD = sorted((ROOT / "experiments/EVAL/k8s/v2/head").glob("job-head-x-*-raunav.yaml"))


@pytest.mark.skipif(not HEAD, reason="no head specs")
def test_each_index_runs_the_script_of_its_spec_under_the_first_specs_template():
    text = R.build("head-x", 3, HEAD, "0" * 40)
    job = yaml.safe_load(text)
    sp = job["spec"]
    assert job["metadata"]["name"] == "head-x-raunav" and sp["completionMode"] == "Indexed"
    assert sp["completions"] == len(HEAD) and sp["parallelism"] == 3 and "backoffLimit" not in sp
    assert all(r["action"] != "FailJob" for r in sp["podFailurePolicy"]["rules"])
    assert R.template_of(job) == R.template_of(yaml.safe_load(HEAD[0].read_text()))
    script = sp["template"]["spec"]["containers"][0]["args"][0]
    assert subprocess.run(["bash", "-n"], input=script, text=True).returncode == 0
    listed = script.split("SPECS=(")[1].split(")")[0].split()
    assert listed == [str(p.relative_to(ROOT)) for p in HEAD]
    assert "exec bash /workspace/script.sh" in script and "git checkout -q " + "0" * 40 in script


def test_specs_with_another_pod_template_are_refused(tmp_path):
    d = yaml.safe_load(HEAD[0].read_text())
    d["spec"]["template"]["spec"]["containers"][0]["resources"]["limits"]["memory"] = "1Gi"
    other = tmp_path / "job-other-raunav.yaml"
    other.write_text(yaml.safe_dump(d))
    with pytest.raises(SystemExit, match="another pod template"):
        R.build("x", 1, [HEAD[0], other], "0" * 40)
