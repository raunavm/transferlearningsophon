"""scripts/build_partrepro_jobs.py: the ParT reproduction reads its md5-checked data from the PVC copy."""
import importlib.util
import pathlib
import subprocess

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
_s = importlib.util.spec_from_file_location("build_partrepro_jobs", ROOT / "scripts" / "build_partrepro_jobs.py")
bpr = importlib.util.module_from_spec(_s)
_s.loader.exec_module(bpr)


def test_every_variant_copies_the_pvc_tarball_and_checks_its_md5():
    for v in bpr.VARIANTS:
        _, text = bpr.spec(v, "mtx-test")
        script = yaml.safe_load(text)["spec"]["template"]["spec"]["containers"][0]["args"][0]
        assert "urlretrieve" not in script and f"cp {bpr.DATA_COPY} TopLandscape.tar" in script
        assert script.index("cp ") < script.index(f"{bpr.DATA_MD5}  TopLandscape.tar")
        assert subprocess.run(["bash", "-n"], input=script, text=True).returncode == 0
