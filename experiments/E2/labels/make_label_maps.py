#!/usr/bin/env python3
"""E2 label maps: 188 -> {2, 10-semantic, 10-random, 30, 188}.

Single source of truth is the official Sophon data config
(experiments/E2/data/JetClassII_full_base.yaml, verbatim copy of
jet-universe/sophon@9dd6dd6 data/JetClassII/JetClassII_full.yaml):
  - the 188 class names are parsed from its label-list comment;
  - the 30 coarse groups are obtained by EVALUATING its own reweight-class
    expressions (label_X_QQ ... label_QCD) on jet_label = 0..187, never by
    hand-transcribed ranges.

Granularities (QCD always the last merged class, matching the 188 ordering):
  g188    identity
  g30     Sophon's 30 reweighting groups, in reweight_classes order
  g10sem  10 semantic super-groups of the 30 (topology x lepton content;
          JetClass-I analogy documented per group — C7: analogue, NOT identity)
  g10rand 10 groups, random partition of the 188 with the SAME group sizes as
          g10sem (C6 stratification), seed 20260714, MATERIALIZED here and
          frozen — never regenerate from the seed alone
  g2      signal (0-160) vs QCD (161-187)

Writes label_maps.json (frozen (188,) int arrays) + label_maps_appendix.md
(the C7 appendix table). Rerunning must be byte-identical.
"""
import json
import re
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
BASE_YAML = HERE.parent / "data" / "JetClassII_full_base.yaml"
RAND_SEED = 20260714

# 10-semantic: partition of the 30 Sophon groups. JC-I analogy in comments.
SEM10 = {
    "X_2P_QQ":       ["label_X_QQ"],                                   # ~ Hbb/Hcc/Zqq/Wqq
    "X_2P_gg":       ["label_X_gg"],                                   # ~ Hgg
    "X_2P_leptonic": ["label_X_ll", "label_X_tauhtaul", "label_X_tauhtauh"],
    "X_YY_4Phad":    ["label_X_YY_QQQQ", "label_X_YY_QQgg", "label_X_YY_gggg"],   # ~ H4q
    "X_YY_3Phad":    ["label_X_YY_QQQ", "label_X_YY_QQg", "label_X_YY_Qgg",
                      "label_X_YY_ggg"],                               # ~ Tbqq (3-prong)
    "X_YY_had_l":    ["label_X_YY_QQll", "label_X_YY_QQl", "label_X_YY_Qll",
                      "label_X_YY_ggll", "label_X_YY_ggl", "label_X_YY_gll"],
    "X_YY_had_tau":  ["label_X_YY_QQtauhtaul", "label_X_YY_QQtauhtauh",
                      "label_X_YY_Qtauhtaul", "label_X_YY_Qtauhtauh",
                      "label_X_YY_ggtauhtaul", "label_X_YY_ggtauhtauh",
                      "label_X_YY_gtauhtaul", "label_X_YY_gtauhtauh"],
    "X_YY_QQ_lv":    ["label_X_YY_QQlv", "label_X_YY_QQtaulv"],        # ~ Hqql/Tbl
    "X_YY_QQ_tauhv": ["label_X_YY_QQtauhv"],
    "QCD":           ["label_QCD"],
}


def parse_class_names(text):
    m = re.search(r"a full list of label names: \[(.*?)\]", text, re.S)
    names = [n.strip() for n in m.group(1).replace("\n", " ").split(",")]
    names = [re.sub(r"^#+\s*", "", n) for n in names if n.strip()]
    assert len(names) == 188, f"expected 188 names, got {len(names)}"
    return names


def sophon_groups(cfg):
    """Evaluate the base config's own reweight-class expressions -> 30 groups."""
    jet_label = np.arange(188)
    order = cfg["weights"]["reweight_classes"]
    groups = {}
    for name in order:
        expr = cfg["new_variables"][name]
        mask = eval(expr, {"jet_label": jet_label, "np": np})
        groups[name] = np.where(mask)[0]
    counts = np.zeros(188, int)
    for idx in groups.values():
        counts[idx] += 1
    assert (counts == 1).all(), "30 groups are not an exact partition of 0..187"
    return groups


def build_map(group_lists):
    """[(name, member_indices)] -> ((188,) int map, class names)."""
    lut = np.full(188, -1, int)
    for g, (_, members) in enumerate(group_lists):
        lut[np.asarray(members, int)] = g
    assert (lut >= 0).all()
    return lut, [n for n, _ in group_lists]


def main():
    text = BASE_YAML.read_text()
    cfg = yaml.safe_load(text)
    names188 = parse_class_names(text)
    g30 = sophon_groups(cfg)
    assert names188[0] == "label_X_bb" and names188[187] == "label_QCD_light"
    assert list(g30)[-1] == "label_QCD" and (g30["label_QCD"] == np.arange(161, 188)).all()

    maps = {}

    lut, cls = build_map([(n, g30[n]) for n in g30])
    maps["g30"] = (lut, cls, "Sophon's own 30 reweighting groups (primary source), "
                             "reweight_classes order")

    sem_lists = []
    for sem_name, members in SEM10.items():
        idx = np.sort(np.concatenate([g30[m] for m in members]))
        sem_lists.append((sem_name, idx))
    lut, cls = build_map(sem_lists)
    maps["g10sem"] = (lut, cls, "10 semantic super-groups of the 30 Sophon groups "
                                "(topology x lepton content); JC-I analogue only (C7)")

    sizes = [len(idx) for _, idx in sem_lists]
    perm = np.random.default_rng(RAND_SEED).permutation(188)
    rand_lists, start = [], 0
    for g, size in enumerate(sizes):
        rand_lists.append((f"rand10_{g}", np.sort(perm[start:start + size])))
        start += size
    lut, cls = build_map(rand_lists)
    maps["g10rand"] = (lut, cls, f"random partition, group sizes matched to g10sem "
                                 f"(C6), numpy default_rng seed {RAND_SEED}, FROZEN")

    lut, cls = build_map([("signal", np.arange(0, 161)), ("QCD", np.arange(161, 188))])
    maps["g2"] = (lut, cls, "signal (jet_label 0-160) vs QCD (161-187)")

    lut, cls = build_map([(n, [i]) for i, n in enumerate(names188)])
    maps["g188"] = (lut, cls, "identity (native JetClass-II labels)")

    # refinement chain (blueprint P1): each rung is a strict coarsening of the
    # next — 2 of 10sem, 10sem of 30, 30 of 188. g10rand deliberately breaks
    # nesting (the control). The 30-partition-of-188 assert is in sophon_groups.
    for coarse, fine in (("g2", "g10sem"), ("g10sem", "g30"), ("g30", "g188")):
        c, f = maps[coarse][0], maps[fine][0]
        for g in range(int(f.max()) + 1):
            assert len(set(c[f == g])) == 1, \
                f"{coarse} does not exactly coarsen {fine} (fine class {g})"

    out = {"provenance": {
        "base_config": "jet-universe/sophon@9dd6dd6a261aa6d5fd2e56f015068127b36854f9 "
                       "data/JetClassII/JetClassII_full.yaml",
        "generator": "experiments/E2/labels/make_label_maps.py",
        "rand_seed": RAND_SEED,
        "note": "maps are FROZEN as materialized arrays; this file is the source of "
                "truth, not the generator seed",
    }, "class_names_188": names188}
    for k, (lut, cls, desc) in maps.items():
        out[k] = {"num_classes": len(cls), "description": desc,
                  "class_names": cls, "map": lut.tolist()}
    (HERE / "label_maps.json").write_text(json.dumps(out, indent=1) + "\n")

    lines = ["# E2 label-merge appendix (C7)", "",
             "Merged-class definitions for every pretraining granularity. Member names",
             "are the native JetClass-II 188 labels (primary source: Sophon data config).",
             ""]
    for k in ("g2", "g10sem", "g10rand", "g30"):
        lut, cls, desc = maps[k]
        lines += [f"## {k} — {len(cls)} classes", "", desc, "",
                  "| idx | merged class | n | members |", "|---|---|---|---|"]
        for g, cname in enumerate(cls):
            members = [names188[i] for i in np.where(lut == g)[0]]
            lines.append(f"| {g} | {cname} | {len(members)} | "
                         + ", ".join(m.replace("label_", "") for m in members) + " |")
        lines.append("")
    lines += ["## g188 — identity", "",
              "| idx | name |", "|---|---|"]
    lines += [f"| {i} | {n} |" for i, n in enumerate(names188)]
    (HERE / "label_maps_appendix.md").write_text("\n".join(lines) + "\n")

    for k, (lut, cls, _) in maps.items():
        binc = np.bincount(lut, minlength=len(cls))
        print(f"{k:8s} {len(cls):3d} classes  sizes={binc.tolist() if len(cls) <= 30 else 'identity'}")
    print("wrote label_maps.json + label_maps_appendix.md")


if __name__ == "__main__":
    main()
