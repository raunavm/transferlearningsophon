#!/usr/bin/env python3
"""Generate the G-campaign Delphes mu-variant cards (PLAN v5.1 §7.1/§7.5).

From the RELEASE production card (delphes_card_CMS_JetClassII.tcl in the
jetclass2_generation repo):

  mu50    : byte-identical COPY of the release card (sha256 hash-compared and
            recorded — §7.1 requires identity, so nothing may be edited, not
            even a RandomSeed line).
  mu0,25,80,140 : differ from the release card ONLY in the `set MeanPileUp`
            line (the stored diff is exactly that one line). mu0 == §7.5
            config (ii): PUPPI runs with zero overlay.
  mu0_nopu : §7.5 config (iii), "pileup module removed" — implemented as
            FUNCTIONAL removal: MeanPileUp 0 + ZVertexSpread 0 + TVertexSpread 0
            + unit vertex formula (no overlay, no hard-scatter vertex smearing).
            LITERAL module removal is not possible in this card: PileUpMerger
            is the sole producer of the vertex collections consumed by the
            PUPPI PV input (`set PVInputArray PileUpMerger/vertices`) and the
            Vertex tree branch, so deleting it breaks PUPPI. This adaptation is
            a pilot-review item (G1); the §7.8 test ((iii) consistent with (i))
            is what validates it.

§7.5 config (i) (raw constituents) is NOT a card: it is the same mu0 run read
through the raw pre-PUPPI EFlow branches (ntupler extension, §7.1.3).

Writes: <out>/delphes_card_CMS_JetClassII_mu{50,0,25,80,140,0_nopu}.tcl
        <out>/card_manifest.json   (sha256 per card + unified diffs vs release)

Usage:
  python3 experiments/G/make_mu_cards.py \
      --release-card /path/to/jetclass2_generation/delphes_cards/delphes_card_CMS_JetClassII.tcl \
      --out experiments/G/cards
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
from pathlib import Path

MUS = [0, 25, 80, 140]


def set_mean_pileup(text: str, mu: int) -> str:
    out, n = re.subn(r"^([ \t]*set MeanPileUp )\d+[ \t]*$",
                     rf"\g<1>{mu}", text, flags=re.M)
    assert n == 1, f"expected exactly one MeanPileUp line, found {n}"
    return out


def make_nopu(text: str) -> str:
    """Functional removal of pileup: no overlay, no vertex smearing."""
    t = set_mean_pileup(text, 0)
    for pat, rep in [
        (r"^([ \t]*set ZVertexSpread )\S+[ \t]*$", r"\g<1>0.0"),
        (r"^([ \t]*set TVertexSpread )\S+[ \t]*$", r"\g<1>0.0"),
        (r"^([ \t]*set VertexDistributionFormula ).*$", r"\g<1>{1.0}"),
    ]:
        t, n = re.subn(pat, rep, t, flags=re.M)
        assert n == 1, f"pattern {pat!r} matched {n} times (expected 1)"
    return t


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def udiff(a: str, b: str, name: str) -> str:
    return "".join(difflib.unified_diff(
        a.splitlines(keepends=True), b.splitlines(keepends=True),
        "release", name))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-card", required=True)
    ap.add_argument("--out", default="experiments/G/cards")
    args = ap.parse_args()
    release = Path(args.release_card).read_text()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    manifest = {"release_card_sha256": sha(release),
                "release_card_source": str(args.release_card), "cards": {}}

    def emit(tag, text):
        p = out / f"delphes_card_CMS_JetClassII_mu{tag}.tcl"
        p.write_text(text)
        d = udiff(release, text, p.name)
        manifest["cards"][p.name] = {
            "sha256": sha(text), "byte_identical_to_release": text == release,
            "diff_lines": sum(1 for l in d.splitlines()
                              if l[:1] in "+-" and l[:3] not in ("+++", "---")),
            "diff": d}
        print(f"  {p.name}: {'IDENTICAL to release' if text == release else str(manifest['cards'][p.name]['diff_lines']) + ' changed lines'}")

    emit("50", release)                      # byte-identical, hash-compared
    for mu in MUS:
        emit(str(mu), set_mean_pileup(release, mu))
    emit("0_nopu", make_nopu(release))

    assert manifest["cards"]["delphes_card_CMS_JetClassII_mu50.tcl"]["byte_identical_to_release"]
    for mu in MUS:
        assert manifest["cards"][f"delphes_card_CMS_JetClassII_mu{mu}.tcl"]["diff_lines"] == 2  # -old +new
    (out / "card_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {out}/card_manifest.json  (release sha {manifest['release_card_sha256'][:16]}…)")


if __name__ == "__main__":
    main()
