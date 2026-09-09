#!/usr/bin/env python3
"""Which published Sophon-family discriminants survive each rung of the ladder.

docs/PRD_PLAN.md 3.1(c): "a table of which published Sophon-family discriminants
are constructible at each of the eight rungs. Property 1 (class-division) makes
this exact."

THE CRITERION, and why it is exact rather than a heuristic. Sophon's Property 1
(arXiv:2405.12972, "class division property") states that a coarse classifier's
score for a merged class is the SUM of the fine scores it absorbed:
g_c = sum_l g_{c_l}. So an analyst holding a rung-R model observes exactly the
group sums, and the linear combinations they can build are

    sum_G a_G g_G  =  sum_i a_{G(i)} g_i,

i.e. precisely those whose per-class coefficient is CONSTANT ON EVERY R-GROUP. A
discriminant -- a ratio of two such combinations -- is constructible at R iff its
numerator and denominator coefficient vectors are each group-constant. No
approximation, no training, no data: it is a statement about the partition.

This is the argument that coarse vocabularies lose USE CASES independently of
transfer quality. A rung does not merely score worse on a discriminant it cannot
express; the nodes the analyst would sum over have stopped existing.

THIS TABLE AND experiments/EVAL/probe.py ANSWER DIFFERENT QUESTIONS, and their
rungs differ for D_bc. Do not "fix" either to agree with the other.

  probe.py::derive_collapsed_at asks: can a rung-R model still be TRAINED to
  separate the signal set from the background set? It fails only when a group
  lands in both -- a third class straddling in from outside is irrelevant,
  because the probe simply never sees those jets.

  This script asks: can the PUBLISHED discriminant be BUILT from output nodes?
  That additionally needs each set to be a union of groups.

  D_bc is the worked case. Its denominator names label_X_cs but not label_X_cq,
  and R63_Q1 merges the two into 2P_HAD_2PARTON|nb0_nc1. The analyst who wants
  g_cs holds only g_{cs+cq}, so what they can build is a different discriminant
  -- D_bc is gone at R63_Q1. The trainable probe survives one rung further, to
  R42_Q1, where label_X_bc finally joins label_X_bq in 2P_HAD_2PARTON|B.
  tests/test_usecase_survival.py pins both numbers and this reason.
"""
import csv
import json
import pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent
MAP = REPO / "configs" / "labelmaps" / "rung_label_maps.v1.csv"
SPEC = REPO / "configs" / "labelmaps" / "usecase_discriminants.v1.csv"

# Contraction order, finest first. docs/DECISIONS.md D3.
RUNGS = ["L188", "L162", "R63_Q1", "R42_Q1", "R29_Q1", "R16_Q1", "R3_VIS", "R1_Q1"]


def read_map():
    with MAP.open() as f:
        rows = list(csv.DictReader(f))
    by_name = {r["class_name"]: int(r["jet_label"]) for r in rows}
    groups = {}
    for rung in RUNGS:
        if rung not in rows[0]:
            raise SystemExit(f"FATAL: rung {rung} missing from {MAP}")
        g = {}
        for r in rows:
            g.setdefault(r[rung], []).append(int(r["jet_label"]))
        groups[rung] = g
    return by_name, groups


def read_spec(by_name):
    """Coefficient vectors per discriminant. `@QCD` and `expand=each` resolved here."""
    with SPEC.open() as f:
        rows = list(csv.DictReader(l for l in f if not l.startswith("#")))
    qcd = sorted(i for n, i in by_name.items() if n.startswith("label_QCD_"))
    if len(qcd) != 27:
        raise SystemExit(f"FATAL: expected 27 QCD classes in the map, found {len(qcd)}")

    def resolve(token):
        if token == "@QCD":
            return list(qcd)
        out = []
        for name in token.split(";"):
            if name not in by_name:
                raise SystemExit(f"FATAL: {name!r} is not a class in {MAP}")
            out.append(by_name[name])
        return out

    specs, titles, sources = {}, {}, {}
    for r in rows:
        d = r["discriminant"]
        titles.setdefault(d, r["title"])
        sources.setdefault(d, r["source"])
        if titles[d] != r["title"] or sources[d] != r["source"]:
            raise SystemExit(f"FATAL: {d} carries two different titles/sources")
        idx = resolve(r["classes"])
        vectors = specs.setdefault(d, {})
        if r["expand"] == "each":
            # Every listed class must be resolvable ON ITS OWN: one indicator
            # vector each. GN3X reports rejection PER QCD subclass, so a rung
            # that merges any two of them cannot produce the row at all.
            for i in idx:
                vectors[f"{r['part']}:{i}"] = {i: float(r["coeff"])}
        else:
            v = vectors.setdefault(r["part"], {})
            for i in idx:
                v[i] = v.get(i, 0.0) + float(r["coeff"])
    return specs, titles, sources


def group_constant(coeff, groups_at_rung):
    """Realisable at this rung iff the coefficient is constant on every group."""
    for members in groups_at_rung.values():
        if len({coeff.get(i, 0.0) for i in members}) > 1:
            return False
    return True


def main():
    by_name, groups = read_map()
    specs, titles, sources = read_spec(by_name)

    table, out = {}, {}
    for d, vectors in specs.items():
        row = {}
        for rung in RUNGS:
            row[rung] = all(group_constant(v, groups[rung]) for v in vectors.values())
        table[d] = row
        alive = [r for r in RUNGS if row[r]]
        out[d] = {
            "title": titles[d],
            "source": sources[d],
            "constructible": row,
            "n_coefficient_vectors": len(vectors),
            # The first rung at which it stops being constructible. None = survives
            # the whole ladder.
            "dies_at": next((r for r in RUNGS if not row[r]), None),
            "last_rung_alive": alive[-1] if alive else None,
        }

    print("| discriminant | " + " | ".join(RUNGS) + " | dies at |")
    print("|---|" + "---|" * (len(RUNGS) + 1))
    for d in specs:
        marks = " | ".join("yes" if table[d][r] else "--" for r in RUNGS)
        print(f"| {titles[d]} | {marks} | {out[d]['dies_at'] or 'survives'} |")

    dest = REPO / "configs" / "labelmaps" / "usecase_survival.v1.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {dest.relative_to(REPO)}")


if __name__ == "__main__":
    main()
