"""The manuscript may not contain a number the generator did not produce.

docs/PLAN.md block G and the project plan both promise this: "The table
generator is the only writer of manuscript numbers; a test fails if the
manuscript contains a number it did not produce." Without it, the provenance
chain stops at `results_generated.tex` -- every macro in that file carries its
source file, the path inside it and a sha256, and then a person types "about
1300" into a sentence and none of that machinery is load-bearing any more.

The rule enforced here: every numeric literal in hand-written manuscript prose
is either (a) a `\\ProbeFoo`-style macro from the generator, or (b) listed in
ALLOWED below with a reason. Generated files are exempt because they ARE the
provenance; they are checked by tests/test_make_tables.py instead.

Deliberately a literal scan rather than a LaTeX parse. A parser would need to
be right about `\\newcommand`, `siunitx` and verbatim to be trusted, and a scan
that is occasionally too strict costs one line in ALLOWED, while a parser that
is occasionally too lax costs the thing the test exists to prevent.
"""
import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
JOURNAL = REPO / "paper" / "journal"

# Files the generator writes. Their numbers ARE the provenance record.
GENERATED = {"results_generated.tex"}

# Numbers allowed in prose, each with the reason it is not a result.
# A number earns a line here only if it cannot change when a job finishes.
ALLOWED = {
    "0": "LaTeX lengths and counters",
    "1": "LaTeX lengths, counters, and \\linewidth fractions",
    "2": "column counts in table preambles",
    "5": "column counts in table preambles",
    "10": "font sizes and \\pt lengths",
    "11": "font sizes",
    "12": "font sizes",
    "95": "the confidence level, fixed by docs/STATISTICS.md, not measured",
}

# Contexts whose digits are never claims about data.
STRIP = [
    (re.compile(r"(?<!\\)%.*$", re.M), ""),                 # comments
    (re.compile(r"\\(?:label|ref|eqref|cite[a-z]*|input|include|usepackage"
                r"|documentclass|bibliographystyle|bibliography|url|href"
                r"|includegraphics|newcommand|renewcommand|def)\s*"
                r"(?:\[[^\]]*\])?\s*(?:\{[^{}]*\})+", re.S), " "),
    (re.compile(r"\\begin\{(?:tabular|array)\}\{[^}]*\}"), " "),
    (re.compile(r"\\[a-zA-Z]+\s*\{[^{}]*\}\s*=\s*[-0-9.]+\s*(?:pt|em|ex|cm|mm|in)"), " "),
    (re.compile(r"-?[0-9.]+\s*(?:pt|em|ex|cm|mm|in|bp|sp)\b"), " "),  # lengths
]

NUMBER = re.compile(r"(?<![A-Za-z\\])-?\d[\d,]*(?:\.\d+)?")


def manuscript_files() -> list[pathlib.Path]:
    if not JOURNAL.is_dir():
        return []
    return sorted(p for p in JOURNAL.rglob("*.tex")
                  if p.name not in GENERATED and "tables" not in p.parts)


def prose_of(path: pathlib.Path) -> str:
    text = path.read_text()
    for pattern, repl in STRIP:
        text = pattern.sub(repl, text)
    return text


def test_the_generated_macros_exist_at_all():
    """If this fails, run experiments/FIGS/make_tables.py -- the rest of this
    file would otherwise pass vacuously."""
    gen = JOURNAL / "results_generated.tex"
    assert gen.is_file(), "no results_generated.tex; the table generator has not run"
    assert gen.read_text().count("newcommand") > 100


@pytest.mark.parametrize("path", manuscript_files() or [None])
def test_no_hand_typed_number_in_the_manuscript(path):
    if path is None:
        pytest.skip("no hand-written manuscript yet; the guard binds when there is one")
    bad = []
    for line_no, line in enumerate(prose_of(path).splitlines(), 1):
        for m in NUMBER.finditer(line):
            if m.group().lstrip("-") in ALLOWED:
                continue
            bad.append(f"{path.relative_to(REPO)}:{line_no}: {m.group()!r} in {line.strip()[:90]!r}")
    assert not bad, (
        "numbers in the manuscript that the table generator did not produce:\n  "
        + "\n  ".join(bad)
        + "\n\nUse a macro from paper/journal/results_generated.tex, or add the "
          "literal to ALLOWED in this file with the reason it cannot change.")


@pytest.mark.parametrize("path", manuscript_files() or [None])
def test_every_macro_the_manuscript_uses_is_one_the_generator_defines(path):
    """The other half of the same guarantee. A macro that is not defined would
    fail at LaTeX time anyway, but a macro that is defined SOMEWHERE ELSE --
    by hand, in the preamble -- would silently reintroduce a typed number."""
    if path is None:
        pytest.skip("no hand-written manuscript yet")
    gen = (JOURNAL / "results_generated.tex").read_text()
    defined = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", gen))
    local = set(re.findall(r"\\(?:new|renew|provide)command\{?\\([A-Za-z]+)\}?",
                           path.read_text()))
    used = set(re.findall(r"\\([A-Za-z]+)", prose_of(path)))
    result_like = {m for m in used
                   if m.startswith(("Probe", "Test", "Recovery", "Bench", "Anomaly",
                                    "Mass", "Aoj", "Survival", "Leg"))}
    undefined = sorted(result_like - defined - local)
    assert not undefined, (
        f"{path.relative_to(REPO)} uses result macros the generator does not "
        f"define: {undefined}. Regenerate, or correct the name.")


def test_the_guard_would_catch_a_typed_number(tmp_path):
    """A guard nobody has seen fail is a guard nobody should trust."""
    f = tmp_path / "fake.tex"
    f.write_text("The rejection reaches 1320 at the headline point.\n")
    found = [m.group() for line in prose_of(f).splitlines()
             for m in NUMBER.finditer(line) if m.group().lstrip("-") not in ALLOWED]
    assert found == ["1320"]


def test_the_guard_does_not_fire_on_a_macro_or_a_comment(tmp_path):
    f = tmp_path / "fake.tex"
    f.write_text("Rejection reaches \\ProbeRejBvcResonantLinearOneeighteight{} here.\n"
                 "% a comment mentioning 1320 must not trip it\n"
                 "\\includegraphics[width=0.8\\linewidth]{figures/probe_ladder_granularity.pdf}\n"
                 "\\cite{Qu2024Sophon}\n")
    found = [m.group() for line in prose_of(f).splitlines()
             for m in NUMBER.finditer(line) if m.group().lstrip("-") not in ALLOWED]
    assert found == [], found
