#!/usr/bin/env python3
"""Generate the G-campaign Delphes mu-variant cards (PLAN v5.1 §7.1/§7.5).

From the RELEASE production card (delphes_card_CMS_JetClassII.tcl in the
jetclass2_generation repo):

  mu50    : PHYSICS-IDENTICAL to the release card (amendment A1,
            registry/gates/plan_amendments.json): the identity hash is computed
            over the card with the TreeWriter output block EXCLUDED. All cards
            (mu50 included) receive two output-only TreeWriter branch additions
            (configs/g/branch_schema.yaml card_requirements) that persist the
            pre-PUPPI EFlow collection and the pileup gen record — without them
            truth references dangle and pu_fraction is uncomputable (§7.7c).
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


OUTPUT_BRANCHES = (
    "  add Branch EFlowMerger/eflow EFlowCandidate ParticleFlowCandidate\n"
    "  add Branch PileUpMerger/stableParticles PileupStableParticle GenParticle\n")
ANCHOR = "  add Branch RunPUPPI/PuppiParticles ParticleFlowCandidate ParticleFlowCandidate\n"


def add_output_branches(text: str) -> str:
    """Insert the two schema-required TreeWriter branches (output-only)."""
    assert text.count(ANCHOR) == 1, "TreeWriter anchor line not found exactly once"
    return text.replace(ANCHOR, ANCHOR + OUTPUT_BRANCHES)


def physics_section(text: str) -> str:
    """Card text with the TreeWriter module block removed (amendment A1)."""
    out, n = re.subn(r"^module TreeWriter TreeWriter \{.*?^\}", "",
                     text, flags=re.M | re.S)
    assert n == 1, f"expected exactly one TreeWriter block, found {n}"
    return out


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
                "release_physics_sha256": sha(physics_section(release)),
                "release_card_source": str(args.release_card),
                "amendment": "A1: identity = physics-section hash (TreeWriter excluded); "
                             "output-only branch additions applied to ALL cards",
                "cards": {}}

    def emit(tag, text):
        text = add_output_branches(text)
        p = out / f"delphes_card_CMS_JetClassII_mu{tag}.tcl"
        p.write_text(text)
        d = udiff(release, text, p.name)
        phys_ok = sha(physics_section(text)) == manifest["release_physics_sha256"]
        manifest["cards"][p.name] = {
            "sha256": sha(text),
            "physics_identical_to_release": phys_ok,
            "diff_lines": sum(1 for l in d.splitlines()
                              if l[:1] in "+-" and l[:3] not in ("+++", "---")),
            "diff": d}
        print(f"  {p.name}: physics_identical={phys_ok}  "
              f"{manifest['cards'][p.name]['diff_lines']} diff lines vs release")

    emit("50", release)                      # byte-identical, hash-compared
    for mu in MUS:
        emit(str(mu), set_mean_pileup(release, mu))
    emit("0_nopu", make_nopu(release))

    # mu50: physics-identical to release (A1), diff = exactly the 2 added output lines
    m50 = manifest["cards"]["delphes_card_CMS_JetClassII_mu50.tcl"]
    assert m50["physics_identical_to_release"] and m50["diff_lines"] == 2
    # mu variants: physics differs ONLY via MeanPileUp; diff = 2 added + MeanPileUp -/+
    for mu in MUS:
        c = manifest["cards"][f"delphes_card_CMS_JetClassII_mu{mu}.tcl"]
        assert not c["physics_identical_to_release"] and c["diff_lines"] == 4
        card_text = (out / f"delphes_card_CMS_JetClassII_mu{mu}.tcl").read_text()
        assert sha(physics_section(card_text)) == sha(physics_section(set_mean_pileup(release, mu)))
    # nopu: the 4-substitution functional removal + 2 added output lines
    assert manifest["cards"]["delphes_card_CMS_JetClassII_mu0_nopu.tcl"]["diff_lines"] == 10
    (out / "card_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {out}/card_manifest.json  (release sha {manifest['release_card_sha256'][:16]}…)")


if __name__ == "__main__":
    main()
