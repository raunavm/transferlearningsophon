"""Every script a job spec runs must parse under the image's Python, 3.10.12.

Python 3.12 (PEP 701) lets an f-string's replacement field hold a backslash, a
comment or the enclosing quote character; 3.10 rejects all three with a
SyntaxError, so such a script passes every local test under 3.13 and dies in the
pod at import. experiments/DIAG/head_epoch_diag.py:446 did exactly that (86d96f5).
ast.parse(feature_version=(3, 10)) does not catch it -- the f-string grammar is not
versioned -- so the f-string rule is checked on the tokens, and the rest of the
grammar with feature_version.

Scope: the scripts on a `python3 <path>.py` line of any experiments/*/k8s spec,
and, recursively, the repository files they load (a "<dir>/<name>.py" literal or an
`experiments.`/`src.`/`scripts.` import)."""
import ast
import io
import pathlib
import re
import sys
import tokenize

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from test_spec_pins import _executed_scripts  # noqa: E402

pytestmark = pytest.mark.skipif(sys.version_info < (3, 12),
                                reason="the f-string token rule needs Python >= 3.12's tokenizer")


def fstring_violations(src: str) -> list[str]:
    """Pre-3.12 f-string rules broken in `src`: a backslash, a comment or the
    enclosing quote character inside a replacement field."""
    out, stack = [], []          # stack of (quote char, depth of braces) per open f-string
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.FSTRING_START:
            stack.append(tok.string.lstrip("rRfFbB")[0])
            continue
        if tok.type == tokenize.FSTRING_END:
            stack.pop()
            continue
        if not stack or tok.type == tokenize.FSTRING_MIDDLE:
            continue
        where = f"line {tok.start[0]}"
        if tok.type == tokenize.COMMENT:
            out.append(f"{where}: comment inside an f-string field")
        elif "\\" in tok.string and tok.type != tokenize.FSTRING_START:
            out.append(f"{where}: backslash inside an f-string field ({tok.string!r})")
        elif tok.type == tokenize.STRING and tok.string.lstrip("rRbBuU")[:1] == stack[-1]:
            out.append(f"{where}: the enclosing quote reused inside an f-string field")
    return out


def _loaded(path: pathlib.Path) -> set[pathlib.Path]:
    src = path.read_text()
    found = set()
    for rel in re.findall(r'["\']((?:experiments|scripts|src)/[\w/]+\.py)["\']', src):
        found.add(ROOT / rel)
    for mod in re.findall(r"^\s*(?:from|import)\s+((?:experiments|src|scripts)(?:\.\w+)+)", src, re.M):
        p = ROOT / (mod.replace(".", "/") + ".py")
        found.add(p if p.exists() else ROOT / mod.replace(".", "/") / "__init__.py")
    return {p for p in found if p.exists()}


def image_scripts() -> list[pathlib.Path]:
    todo = {ROOT / s for spec in ROOT.glob("experiments/*/k8s/*.yaml")
            for s in _executed_scripts(spec.read_text())}
    todo = {p for p in todo if p.exists()}
    seen = set()
    while todo:
        p = todo.pop()
        seen.add(p)
        todo |= _loaded(p) - seen
    return sorted(seen)


def test_the_token_rule_catches_what_310_rejects():
    assert fstring_violations("x = f\"{'a' + '\\n'}\"\n")
    assert fstring_violations('x = f"{d["k"]}"\n')
    assert fstring_violations("x = f'{d[\"k\"]}'\n") == []
    assert fstring_violations("x = f\"{v!r:>10}\\n\"\n") == []       # backslash outside the field


def test_the_scope_is_not_vacuous():
    s = {p.relative_to(ROOT).as_posix() for p in image_scripts()}
    assert {"experiments/EVAL/extract_v2.py", "experiments/EVAL/extract_features.py",
            "experiments/EVAL/anomaly.py"} <= s


@pytest.mark.parametrize("path", image_scripts(), ids=lambda p: p.relative_to(ROOT).as_posix())
def test_every_script_a_spec_runs_parses_under_python_310(path):
    src = path.read_text()
    ast.parse(src, filename=str(path), feature_version=(3, 10))
    assert fstring_violations(src) == [], f"{path.relative_to(ROOT)}: {fstring_violations(src)}"
