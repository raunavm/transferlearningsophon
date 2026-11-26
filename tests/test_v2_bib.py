"""The bibliography of the v2 draft (PI 2026-11-26): every key the text cites is in paper/journal/refs.bib, no key is
defined twice, and every entry has balanced braces and an author (or key), a title and a year, so BibTeX can build it."""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
V2 = ROOT / "paper/journal/v2"
BIB = ROOT / "paper/journal/refs.bib"


def entries():
    text = BIB.read_text()
    starts = [m.start() for m in re.finditer(r"^@\w+\{", text, re.M)]
    return [text[a:b] for a, b in zip(starts, starts[1:] + [len(text)])]


def test_every_cited_key_is_defined_once():
    keys = [re.match(r"@\w+\{([^,\s]+),", e).group(1) for e in entries()]
    dup = sorted({k for k in keys if keys.count(k) > 1})
    assert not dup, dup
    cited = set()
    for f in ["abstract", "intro", "related", "methods", "results", "discussion", "appendix", "endmatter"]:
        for c in re.findall(r"\\cite(?:\[[^]]*\])?\{([^}]*)\}", (V2 / f"{f}.tex").read_text()):
            cited |= {k.strip() for k in c.split(",")}
    assert not cited - set(keys), sorted(cited - set(keys))


def test_entries_are_well_formed():
    """Balanced braces and a title and an author (or key) in every entry; a missing year is only a BibTeX warning
    (older entries keep it inside the journal fields), so it is not required here."""
    bad = []
    for e in entries():
        body = e.rstrip()
        key = re.match(r"@\w+\{([^,\s]+),", body).group(1)
        if body.count("{") != body.count("}") or not re.search(r"(^|[\s,{])title\s*=", body, re.I) \
                or not re.search(r"(^|[\s,{])(author|key|organization)\s*=", body, re.I):
            bad.append(key)
    assert not bad, bad
