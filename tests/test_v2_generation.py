"""The checkpoint's table and macro generation as the render runs it (PI 2026-11-26): mlst_v2_tables.py and
perclass_jc1_table.py at this commit, so that a build stop (FATAL) appears in the GT-8 gate and not in the render."""
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_v2_tables_and_macros_build():
    for script in ("experiments/FIGS/mlst_v2_tables.py", "experiments/FIGS/perclass_jc1_table.py"):
        p = subprocess.run([sys.executable, script], cwd=ROOT, capture_output=True, text=True)
        assert p.returncode == 0, f"{script}:\n{(p.stdout + p.stderr)[-3000:]}"
    assert (ROOT / "paper/journal/v2/tables/numbers.tex").exists()
