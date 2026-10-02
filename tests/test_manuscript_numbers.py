"""The manuscript may not contain a number the generator did not produce.

docs/PLAN.md block G and the project plan both promise this: "The table
generator is the only writer of manuscript numbers; a test fails if the
manuscript contains a number it did not produce." Without it, the provenance
chain stops at `results_generated.tex` -- every macro in that file carries its
source file, the path inside it and a sha256, and then a person types "about
1300" into a sentence and none of that machinery is load-bearing any more.

The rule enforced here: every numeric literal in hand-written manuscript prose,
digits or a number word ("five", "a quarter"), is either (a) a `\\ProbeFoo`-style
macro from the generator, or (b) listed in ALLOWED (digits) or ALLOWED_WORDS
(phrases) below with a reason. Generated files are exempt because they ARE the
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
# Files the generator writes, as paths relative to paper/journal. Their numbers
# ARE the provenance record. Paths, not directories: exempting all of tables/
# would exempt a hand-written table dropped in there, and nothing would notice.
GENERATED = {"results_generated.tex",
             "tables/probes_linear.tex",
             "tables/probes_mlp.tex",
             "tables/usecase_survival.tex",
             "tables/finetuning_wave1.tex",
             "tables/label_recovery.tex",
             "tables/random_control.tex",
             "tables/finetune.tex",
             "tables/finetune_accuracy.tex",
             "tables/anomaly.tex",
             "tables/mass.tex",
             "tables/realdata.tex",
             "tables/finetune_recipe.tex",
             "tables/appendix_levels.tex",
             "tables/appendix_vocabulary.tex",
             "tables/appendix_tasks.tex",
             "tables/anomaly_per_run.tex"}

# Numbers allowed in prose, each with the reason it is not a result.
# A number earns a line here only if it cannot change when a job finishes.
ALLOWED = {
    "0": "LaTeX lengths and counters",
    "1": "LaTeX lengths, counters, \\linewidth fractions, and the 1 of a definition "
         "(1-AUC, x/(1+x), +-1 standard deviation)",
    "2": "column counts in table preambles",
    "5": "column counts in table preambles",
    "10": "font sizes and \\pt lengths",
    "11": "font sizes",
    "12": "font sizes",
    "95": "the confidence level, fixed by docs/STATISTICS.md, not measured",
    "68": "the area inside sigma_eff, its definition, fixed by the PRESPEC S7 amendment",
}

# Contexts whose digits are never claims about data.
STRIP = [
    (re.compile(r"(?<!\\)%.*$", re.M), ""),                 # comments
    (re.compile(r"\\(?:label|ref|eqref|cite[a-z]*|input|include|usepackage"
                r"|documentclass|bibliographystyle|bibliography|url|href"
                r"|includegraphics|newcommand|renewcommand|def)\s*"
                r"(?:\[[^\]]*\])?\s*(?:\{[^{}]*\})+", re.S), " "),
    # a column spec may nest one level of braces: l p{0.45\linewidth}
    (re.compile(r"\\begin\{(?:tabular|array)\}\{(?:[^{}]|\{[^{}]*\})*\}"), " "),
    (re.compile(r"\\[a-zA-Z]+\s*\{[^{}]*\}\s*=\s*[-0-9.]+\s*(?:pt|em|ex|cm|mm|in)"), " "),
    (re.compile(r"-?[0-9.]+\s*(?:pt|em|ex|cm|mm|in|bp|sp)\b"), " "),  # lengths
]

NUMBER = re.compile(r"(?<![A-Za-z\\])-?\d[\d,]*(?:\.\d+)?")

# A count spelled out is as much a typed number as its digits: "pretrained five
# times" goes stale exactly like "pretrained 5 times" when a row has two runs.
# The generator emits the word forms the prose needs (\ProbeNSeedsWord,
# \RandNDraws, ...). Macro names spell digits too (\VocabSizeLonesixtwo), but
# inside one word, so the word boundaries never match there.
NUMBER_WORD = re.compile(
    r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|"
    r"fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|hundreds?|"
    r"thousands?|quarters?|half|halves|dozens?)\b", re.I)

# Number words that are structure, not a count of anything run or measured, each
# with the reason it cannot change. Matched as PHRASES and removed before the
# scan, so the allowance never extends to the bare word.
ALLOWED_WORDS = [
    (r"\b(?:one|two|three|four)-(?=[^.;]{0,40}?prong)",
     "prong counts name decay topologies (two-prong, three- and four-prong)"),
    (r"\b(?:two|three|four)-quark\b",
     "parton counts name decay topologies (the four-quark decays X->YY->QQQQ)"),
    (r"\binto\s+four\s+quarks\s+or\s+gluons\b",
     "names the decay topology of the left-out family, X->YY->QQQQ with gluons included "
     "(the appendix's 'four-prong: four quarks or gluons' group)"),
    (r"\btwo-by-two\b", "the name of the vocabulary-by-mass-output design"),
    (r"\bleave-one-family-out\b", "the name of the method"),
    (r"\bone-versus-rest\b", "the definition of the macro AUC"),
    (r"\bmax SIC of one\b",
     "an identity: with no discrimination eps_S = eps_B, so eps_S/sqrt(eps_B) <= 1"),
    (r"\bhalf the smallest interval\b", "the definition of sigma_eff"),
]


def without_allowed_words(text: str) -> str:
    """The prose with every ALLOWED_WORDS phrase blanked; applied to the whole text,
    since a phrase can break across lines, and keeping every newline."""
    for pattern, _ in ALLOWED_WORDS:
        text = re.sub(pattern, " ", text, flags=re.I)
    return text


def number_words(line: str) -> list[str]:
    return [m.group() for m in NUMBER_WORD.finditer(line)]


# Macros that hold a ratio of seed means with no error: kept in the generated
# file, never quoted. Every ratio the text quotes is a \Paired... macro, the
# paired geometric mean with its 95 % interval (Sec. 3.5).
RATIO_OF_MEANS = re.compile(r"\\(\w*OmaRatio\w*|Rand(?:Sem)?CostFactor\w*|VcbFactor\w*)")


def manuscript_files() -> list[pathlib.Path]:
    if not JOURNAL.is_dir():
        return []
    return sorted(p for p in JOURNAL.rglob("*.tex")
                  if p.relative_to(JOURNAL).as_posix() not in GENERATED)


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
    prose = prose_of(path)
    for line_no, (line, words) in enumerate(zip(prose.splitlines(),
                                                without_allowed_words(prose).splitlines()), 1):
        for m in NUMBER.finditer(line):
            if m.group().lstrip("-") in ALLOWED:
                continue
            bad.append(f"{path.relative_to(REPO)}:{line_no}: {m.group()!r} in {line.strip()[:90]!r}")
        for w in number_words(words):
            bad.append(f"{path.relative_to(REPO)}:{line_no}: {w!r} in {line.strip()[:90]!r}")
    assert not bad, (
        "numbers in the manuscript that the table generator did not produce:\n  "
        + "\n  ".join(bad)
        + "\n\nUse a macro from paper/journal/results_generated.tex (word forms exist for "
          "the counts the prose spells out), or add the literal to ALLOWED, or the "
          "phrase to ALLOWED_WORDS, in this file with the reason it cannot change.")


@pytest.mark.parametrize("path", manuscript_files() or [None])
def test_no_ratio_of_means_is_quoted(path):
    """A ratio of seed means carries no error and is not the point the paired
    interval belongs to; the text quotes the paired ratio instead."""
    if path is None:
        pytest.skip("no hand-written manuscript yet")
    used = sorted(set(RATIO_OF_MEANS.findall(prose_of(path))))
    assert not used, f"{path.relative_to(REPO)} quotes ratios of means: {used}; use \\Paired..."


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
                                    "Mass", "Aoj", "Survival", "Leg",
                                    "Acc", "Mde", "Pair", "SignAgree", "Tost",
                                    "Trend", "Use", "Vocab", "Design", "Rand", "Ft", "Lit", "Vcb"))}
    # NOT "- local": a result macro defined by hand in this file is precisely the
    # hazard named above, so a local definition aggravates it, never excuses it.
    undefined = sorted((result_like - defined) | (result_like & local))
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


def test_the_guard_would_catch_a_spelled_out_count(tmp_path):
    """Number words are numbers; structure words are allowed only as phrases."""
    f = tmp_path / "fake.tex"
    f.write_text("Each vocabulary is pretrained five times and saw a quarter of the jets.\n"
                 "The two-prong and three- and four-prong decays, a two-by-two design.\n"
                 "\\VocabSizeLonesixtwo{} classes, \\ProbeNSeedsWord{} runs, two runs.\n")
    found = [w for line in without_allowed_words(prose_of(f)).splitlines()
             for w in number_words(line)]
    assert found == ["five", "quarter", "two"], found


def test_the_guard_does_not_fire_on_a_macro_or_a_comment(tmp_path):
    f = tmp_path / "fake.tex"
    f.write_text("Rejection reaches \\ProbeRejBvcResonantLinearOneeighteight{} here.\n"
                 "% a comment mentioning 1320 must not trip it\n"
                 "\\includegraphics[width=0.8\\linewidth]{figures/probe_ladder_granularity.pdf}\n"
                 "\\cite{Qu2024Sophon}\n")
    found = [m.group() for line in prose_of(f).splitlines()
             for m in NUMBER.finditer(line) if m.group().lstrip("-") not in ALLOWED]
    assert found == [], found
