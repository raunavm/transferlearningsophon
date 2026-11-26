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


def test_every_build_macro_the_text_uses_is_defined():
    """Ledger 35: every \\V..., \\Rev..., \\Mlst... macro the v2 text uses is written by the build (run in the test
    above), so a missing number stops the gate, not the render."""
    import re
    tables = ROOT / "paper/journal/v2/tables"
    defined = set()
    for f in ("numbers.tex", "results_generated.tex"):
        p = tables / f if (tables / f).exists() else ROOT / "paper/journal" / f
        defined |= set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", p.read_text()))
    used = set()
    for f in ("abstract", "intro", "related", "methods", "results", "discussion", "appendix", "endmatter"):
        text = re.sub(r"(?<!\\)%.*", "", (ROOT / "paper/journal/v2" / f"{f}.tex").read_text())
        used |= set(re.findall(r"\\((?:V|Rev|Mlst|Design|Lit|Ft|Fill|Lp|JcBal|Xbx|Xbb|Xbs|Rand|K)[A-Za-z]*)", text))
    missing = sorted(u for u in used if u not in defined)
    assert not missing, missing
