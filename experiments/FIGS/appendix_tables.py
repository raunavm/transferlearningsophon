"""The appendix that says what every class, group and task in the paper is.

Called by make_tables.py, which writes its output beside the other generated
tables. Three tables, all read from the files that define them, never typed:

  * appendix_levels.tex -- the vocabularies: the 17 coarsest groups in words, and
    how many classes each level of the tree has and whether it was pretrained;
  * appendix_vocabulary.tex -- every native JetClass-II class with its group at
    each coarser level and in every random partition (configs/labelmaps/
    rung_label_maps.v1.csv, rand_label_map.v*.csv);
  * appendix_tasks.tex -- the native classes of every probe task and anomaly
    signal, taken from the code that ran them (experiments/EVAL/probe.py TASKS,
    anomaly.py SIGNAL_SUITE) and checked against the result files that record
    what ran (the probe ladder's `names`, the anomaly summary's signals).

A disagreement between the code and a result file stops the build: it would mean
the appendix describes a task other than the one the numbers came from.
"""
from __future__ import annotations

import collections
import csv
import importlib.util
import json
import pathlib
import re

# Tree levels, finest first (docs/DECISIONS.md D3); the order columns are printed in.
RUNGS = ["L188", "L162", "R63_Q1", "R42_Q1", "R29_Q1", "R16_Q1", "R3_VIS", "R1_Q1"]
SHOWN = ["R63_Q1", "R42_Q1", "R29_Q1", "R16_Q1"]      # coarser than 162, finer than 4
PRETRAINED = {"L188", "L162", "R42_Q1", "R16_Q1"}
# The 64- and 30-class rerun arms were cut from the grid on 2026-10-09, so the paper
# pretrains no level outside PRETRAINED.
PRETRAINED_V2: set = set()
# Column headings of the random partitions with controlled flavor boundaries, under a
# "flavor boundary" heading.
FLAV_HEAD = {"FLAV_F0": "none", "FLAV_F1": "$b$ quark", "FLAV_F1R": "random"}

_TOKENS = re.compile(r"tauh|taue|taum|[bcsqgemvQ]")
_TOKEN_TEX = {"tauh": r"\tau_h", "taue": r"\tau_e", "taum": r"\tau_\mu", "e": "e",
              "m": r"\mu", "v": r"\nu"}


def native_tex(name: str) -> str:
    """label_X_YY_bctauhv -> $X\\to YY\\to bc\\tau_h\\nu$; label_QCD_bbc -> QCD, $bbc$.
    Also a flavour orbit of flavour_pair.v2.json, with Q for any quark:
    X_YY_QQmm -> $X\\to YY\\to QQ\\mu\\mu$."""
    body = name.removeprefix("label_")
    if body.startswith("QCD_"):
        rest = body[4:]
        return "QCD, light" if rest == "light" else f"QCD, ${rest}$"
    head, _, prods = body.rpartition("_")
    toks = _TOKENS.findall(prods)
    if "".join(toks) != prods:
        raise SystemExit(f"FATAL: cannot read the decay products of {name}")
    chain = r"X\to YY\to " if head == "X_YY" else r"X\to "
    return "$" + chain + "".join(_TOKEN_TEX.get(t, t) for t in toks) + "$"


_PRONG = {"2P": "two-prong", "3P": "three-prong", "4P": "four-prong"}
_PARTONS = {"1PARTON": "one quark or gluon", "2PARTON": "two quarks or gluons",
            "3PARTON": "three quarks or gluons", "4PARTON": "four quarks or gluons"}
_LEPTONS = {"LL": r"$\ell\ell$", "L": r"$\ell$", "TAUH": r"$\tau_h$", "TAUL": r"$\tau_\ell$"}


def group_tex(name: str) -> str:
    """A coarsest-level group name in words: 4P_SEMILEP_NU_2PARTON_L ->
    four-prong: two quarks or gluons, $\\ell$, $\\nu$."""
    if name == "QCD_ALL":
        return "QCD"
    parts = name.split("_")
    words = [_PARTONS[p] for p in parts if p in _PARTONS]
    words += [_LEPTONS[p] for p in parts if p in _LEPTONS]
    words += [r"$\nu$" for p in parts if p == "NU"]
    if parts[0] not in _PRONG or not words:
        raise SystemExit(f"FATAL: cannot describe group {name}")
    return f"{_PRONG[parts[0]]}: " + ", ".join(words)


def read_map(path: pathlib.Path) -> list[dict]:
    with pathlib.Path(path).open() as f:
        return list(csv.DictReader(f))


def partitions(maps_dir: pathlib.Path, rows: list[dict]) -> list[tuple[str, str, dict]]:
    """[(heading, column, {class_name: group id})] from every partition map in
    configs/labelmaps (any *_map*.v*.csv but the tree's own: the random partitions'
    rand_label_map.v*.csv and the flavour pair's flavour_pair_map.v2.csv), each
    file's partitions in column order, random partitions first. The first grid's
    random partitions (v1) belong to the retired first set of runs and are not
    printed. The rerun's (v2) are headed by their number, as in the paper's
    "random partition 3"; a later version's by version and number (v3.2); the
    flavor control label sets by FLAV_HEAD."""
    out = []
    names = {r["class_name"] for r in rows}
    version_of = lambda f: re.search(r"\.v(\d+)\.csv$", f.name).group(1)
    # By version, the random partitions of each version before its other maps.
    for f in sorted(maps_dir.glob("*_map*.v*.csv"),
                    key=lambda f: (int(version_of(f)), not f.name.startswith("rand_"), f.name)):
        if f.name.startswith("rung_label_maps") or version_of(f) == "1":
            continue
        version = version_of(f)
        prows = read_map(f)
        if {r["class_name"] for r in prows} != names:
            raise SystemExit(f"FATAL: {f} does not cover the native classes of the label map")
        cols = [c for c in prows[0] if re.fullmatch(r"(RAND|FLAV)\w*", c) and not c.endswith("_name")]
        for c in cols:
            m = re.fullmatch(r"RAND\d*_p(\d+)", c)
            head = (m.group(1) if m and version == "2" else f"v{version}.{m.group(1)}" if m
                    else FLAV_HEAD.get(c, c.split("_", 1)[-1]))
            out.append((head, c, {r["class_name"]: r[c] for r in prows}))
    return sorted(out, key=lambda p: not p[1].startswith("RAND"))


def check_grid_covered(grid_path: pathlib.Path, parts: list) -> None:
    """Every classification arm of the rerun grid trains on a tree level or on a
    partition the appendix prints; anything else would be a vocabulary the paper
    reports without defining. An arm's vocabulary is its config's name, less the
    mass-output suffixes (the leave-one-family-out arms reuse their parents'
    configs, the self-supervised arm has no classes)."""
    printed = {c for _, c, _ in parts}
    for arm in json.loads(grid_path.read_text())["arms"]:
        if arm.get("num_classes") is None:
            continue
        vocab = re.sub(r"(_MASS)?(_LM)?$", "", pathlib.Path(arm["config"]).stem)
        if vocab not in RUNGS and vocab not in printed:
            raise SystemExit(f"FATAL: {grid_path} trains {arm['name']} on {vocab}, which no "
                             f"partition map in configs/labelmaps prints in the appendix")


def _longtable(colspec: str, head: str, rows: list[str], caption: str, label: str) -> str:
    return "\n".join([
        "{\\footnotesize", "\\setlength{\\tabcolsep}{3pt}",
        f"\\begin{{longtable}}{{{colspec}}}",
        f"\\caption{{{caption}}}\\label{{{label}}}\\\\", "\\toprule", head, "\\midrule",
        "\\endfirsthead", "\\toprule", head, "\\midrule", "\\endhead",
        "\\bottomrule", "\\endlastfoot", *rows, "\\end{longtable}", "}", ""])


def table_vocabulary(rows: list[dict], parts: list) -> str:
    """Every native class, its group at 64, 43, 30 and 17 classes and in each partition."""
    order = {r: {} for r in SHOWN}
    for r in rows:
        for rung in SHOWN:
            order[rung].setdefault(r[rung], len(order[rung]))
    key = lambda r: tuple(order[g][r[g]] for g in ("R16_Q1", "R42_Q1", "R63_Q1")) + (int(r["jet_label"]),)
    sizes = {rung: len({r[rung] for r in rows}) for rung in RUNGS}
    rand = [h for h, c, _ in parts if c.startswith("RAND")]
    flav = [h for h, c, _ in parts if not c.startswith("RAND")]
    head = (f" & \\multicolumn{{{len(SHOWN)}}}{{c}}{{level}}"
            + (f" & \\multicolumn{{{len(rand)}}}{{c}}{{random partition}}" if rand else "")
            + (f" & \\multicolumn{{{len(flav)}}}{{c}}{{flavor boundary}}" if flav else "")
            + " \\\\\n"
            + "JetClass-II class & " + " & ".join(str(sizes[r]) for r in SHOWN)
            + "".join(f" & {h}" for h, _, _ in parts) + " \\\\")
    body, last = [], None
    for r in sorted(rows, key=key):
        if last is not None and r["R16_Q1"] != last:
            body.append("\\addlinespace[2pt]")
        last = r["R16_Q1"]
        body.append(native_tex(r["class_name"]) + " & " + " & ".join(r[g] for g in SHOWN)
                    + "".join(f" & {p[r['class_name']]}" for _, _, p in parts) + " \\\\")
    caption = ("Every JetClass-II class and its group at each coarser level of the class "
               f"hierarchy (columns headed by the number of classes at that level: "
               f"{', '.join(str(sizes[r]) for r in SHOWN)}), in each random partition, numbered as "
               "in Table~\\ref{tab:pairs}, and in each random partition with controlled flavor "
               "boundaries: no flavor boundary, a $b$-quark boundary and a random flavor boundary "
               "(Section~\\ref{sec:controls}). "
               "Group numbers are those of the released label maps; classes that share a number "
               "in a column share a class in that label set. At the "
               f"{sizes['L162']}-class level every resonant class keeps its own label and the "
               f"QCD classes share one; at {sizes['R3_VIS']} classes the groups are decays with "
               f"{visible_prongs(rows)} visible prongs (a neutrino is not counted) and QCD. "
               "Rows are ordered by the "
               f"{sizes['R16_Q1']}-class group (Table~\\ref{{tab:app-levels}}), with a gap between "
               "groups. $q$ is a $u$ or $d$ quark; $\\tau_h$ a hadronic and $\\tau_e$, $\\tau_\\mu$ "
               "leptonic $\\tau$ decays.")
    ncol = len(SHOWN) + len(parts)
    return _longtable("l " + "r" * ncol, head, body, caption, "tab:app-vocabulary")


_WORD = {2: "two", 3: "three", 4: "four"}


def visible_prongs(rows: list[dict]) -> str:
    """The 4-class level in words, from its group names (2P_VIS, 3P_VIS, ...): it
    counts VISIBLE prongs, so a four-body decay with a neutrino sits with the
    three-prong decays."""
    names = {r["R3_VIS_name"] for r in rows} - {"QCD_ALL"}
    k = sorted(int(m.group(1)) for n in names for m in [re.fullmatch(r"(\d)P_VIS", n)] if m)
    if len(k) != len(names) or not k:
        raise SystemExit(f"FATAL: the 4-class level's groups {sorted(names)} are not n-prong "
                         "visible-content groups")
    w = [_WORD[n] for n in k]
    return ", ".join(w[:-1]) + " or " + w[-1] if len(w) > 1 else w[0]


def level_rules(rows: list[dict]) -> dict:
    """What each level of the tree labels, in words, each checked against the
    group names of the label map so that a description cannot drift from it: the
    flavour tag a level's names carry (|nb2_nc0, |B, |HF), none at the 17-class
    level, the visible-prong count at the 4-class level."""
    names = {r: {row[f"{r}_name"] for row in rows} for r in RUNGS}
    qcd = {row["L162"] for row in rows if row["class_name"].startswith("label_QCD_")}
    n_res = sum(not row["class_name"].startswith("label_QCD_") for row in rows)
    tags = {r: {n.split("|", 1)[1] for n in names[r] if "|" in n} for r in RUNGS}
    checks = {
        "L188": len(names["L188"]) == len(rows),
        "L162": len(qcd) == 1 and len(names["L162"]) == n_res + 1,
        "R63_Q1": bool(tags["R63_Q1"]) and all(re.fullmatch(r"nb\d_nc\d", t) for t in tags["R63_Q1"]),
        "R42_Q1": bool(tags["R42_Q1"]) and tags["R42_Q1"] <= {"B", "C", "LG"},
        "R29_Q1": bool(tags["R29_Q1"]) and tags["R29_Q1"] <= {"HF", "LG"},
        "R16_Q1": not tags["R16_Q1"],
        "R3_VIS": bool(visible_prongs(rows)),
        "R1_Q1": names["R1_Q1"] == {"RESONANT_ALL", "QCD_ALL"}}
    bad = [r for r, ok in checks.items() if not ok]
    if bad:
        raise SystemExit(f"FATAL: the appendix's description of {bad} no longer matches the "
                         "group names of the label map")
    return {"L188": "the JetClass-II classes",
            "L162": "one QCD class; every resonant class kept",
            "R63_Q1": "each decay topology by its numbers of $b$ and $c$ quarks",
            "R42_Q1": "each decay topology by whether it holds a $b$ quark, else a $c$, else neither",
            "R29_Q1": "each decay topology by whether it holds a $b$ or $c$ quark",
            "R16_Q1": "the number of prongs and whether each is a quark or gluon (without distinguishing them), an electron or muon, a $\\tau$ lepton or a neutrino, without flavor",
            "R3_VIS": f"{visible_prongs(rows)} visible prongs (a neutrino is not counted), or QCD",
            "R1_Q1": "resonance decay or QCD"}


def table_levels(rows: list[dict], v2_levels: bool) -> str:
    """The levels of the tree, and the coarsest pretrained vocabulary in words."""
    sizes = {rung: len({r[rung] for r in rows}) for rung in RUNGS}
    rule = level_rules(rows)
    pre = lambda r: ("yes" if r in PRETRAINED else
                     ("yes (rerun)" if v2_levels else "\\pending{rerun}") if r in PRETRAINED_V2 else "no")
    body = [f"{sizes[r]} & {rule[r]} & {pre(r)} \\\\" for r in RUNGS]
    body.append("\\midrule")
    groups = collections.OrderedDict()
    for r in rows:
        groups.setdefault((r["R16_Q1"], r["R16_Q1_name"]), []).append(r)
    body.append(f"\\multicolumn{{3}}{{@{{}}l}}{{\\itshape the {sizes['R16_Q1']} classes of the "
                f"{sizes['R16_Q1']}-class label set}} \\\\")
    for (gid, name), members in groups.items():
        body.append(f"{gid} & {group_tex(name)} & {len(members)} {'class' if len(members) == 1 else 'classes'} \\\\")
    caption = ("The levels of the class hierarchy, finest first: the number of classes, what each "
               "level labels, and whether models were pretrained on it. Below, the groups of the "
               f"{sizes['R16_Q1']}-class label set, by the group number used in "
               "Table~\\ref{tab:app-vocabulary}, with the number of JetClass-II classes each merges. "
               "$\\ell$ is an electron or a muon; $\\tau_\\ell$ a leptonic $\\tau$ decay. A four-prong "
               "group with a neutrino has three visible prongs.")
    lines = ["\\begin{table}[p]", "\\centering", "\\small", f"\\caption{{{caption}}}",
             "\\label{tab:app-levels}", "\\begin{tabular}{r p{0.62\\linewidth} l}", "\\toprule",
             "classes & labels & pretrained \\\\", "\\midrule", *body, "\\bottomrule",
             "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def _load_probe(root: pathlib.Path):
    spec = importlib.util.spec_from_file_location("probe_for_appendix",
                                                  root / "experiments" / "EVAL" / "probe.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def table_tasks(root: pathlib.Path, rows: list[dict], ladder: dict, vcb: dict | None,
                anomaly: dict | None, labels: dict) -> str:
    """The native classes of each probe task and anomaly signal, from the code that ran them."""
    probe = _load_probe(root)
    by_label = {int(r["jet_label"]): r["class_name"] for r in rows}
    sizes = {rung: len({r[rung] for r in rows}) for rung in RUNGS}
    lines = []
    ran = dict(ladder["tasks"])
    if vcb:
        ran.update(vcb["tasks"])
    for task, T in ran.items():
        spec = probe.TASKS.get(task)
        if spec is None:
            raise SystemExit(f"FATAL: {task} ran but experiments/EVAL/probe.py does not define it")
        sig = [by_label[i] for i in spec["signal"]]
        bkg = [by_label[i] for i in spec["background"]]
        if len(sig) == 1 and len(bkg) == 1 and T["names"] != sig + bkg:
            raise SystemExit(f"FATAL: {task}: the result file ran {T['names']}, the code defines "
                             f"{sig + bkg}")
        all_qcd = {r["class_name"] for r in rows if r["class_name"].startswith("label_QCD_")}
        if all_qcd <= set(bkg):
            btxt = ", ".join([native_tex(c) for c in bkg if c not in all_qcd]
                             + [f"all {len(all_qcd)} QCD classes"])
        else:
            btxt = ", ".join(native_tex(c) for c in bkg)
        merged = T.get("collapsed_at") or []
        where = str(sizes[merged[0]]) if merged else "never"
        win = spec.get("window")
        wtxt = ""
        if win:
            lo_pt, hi_pt = win["jet_pt"]
            lo_m, hi_m = win["jet_sdmass"]
            wtxt = (f"; ${lo_pt:.0f}<p_{{\\mathrm T}}<{hi_pt:.0f}$~GeV, "
                    f"${lo_m:.0f}<m_{{\\mathrm{{SD}}}}<{hi_m:.0f}$~GeV, "
                    f"$|\\eta|<{win['jet_eta'][1]:g}$")
        lines.append(f"{labels.get(task, task)} & {', '.join(native_tex(c) for c in sig)} & "
                     f"{btxt}{wtxt} & {where} \\\\")
    if anomaly:
        src = (root / "experiments" / "EVAL" / "anomaly.py").read_text()
        m = re.search(r"^SIGNAL_SUITE = \[(.*?)\]", src, re.S | re.M)
        suite = re.findall(r'"(label_\w+)"', m.group(1)) if m else []
        ran_sigs = set(anomaly["families"][next(iter(anomaly["families"]))])
        if set(suite) != ran_sigs:
            raise SystemExit(f"FATAL: anomaly.py SIGNAL_SUITE {suite} is not the signal set the "
                             f"summary holds {sorted(ran_sigs)}")
        group_of = {r["class_name"]: r["R16_Q1"] for r in rows}
        lines.append("\\midrule")
        lines.append("\\multicolumn{4}{@{}l}{\\itshape anomaly signals, injected into QCD jets} \\\\")
        removed = anomaly.get("classes_removed_by_level", {}).get("class_sum_matched", {})
        for s in suite:
            members = [c for c in group_of if group_of[c] == group_of[s]]
            n = set(removed.get(s, {}).values())
            if n and n != {len(members)}:
                raise SystemExit(f"FATAL: the output ratio for {s} left out {n} classes, but its "
                                 f"{sizes['R16_Q1']}-class group holds {len(members)}")
            lines.append(f"anomaly signal & {native_tex(s)} & QCD test jets; the output ratio "
                         f"leaves out the {len(members)} native classes of its "
                         f"{sizes['R16_Q1']}-class group ({group_of[s]}) & --- \\\\")
    caption = ("The native classes of every frozen-probe task (signal, then background) and of "
               "every anomaly signal, and the number of classes of the finest vocabulary in "
               "which the task's two sides share a class. A kinematic window, where given, "
               "applies to both sides.")
    return "\n".join(["\\begin{table}[p]", "\\centering", "\\small", f"\\caption{{{caption}}}",
                      "\\label{tab:app-tasks}", "\\begin{adjustbox}{max width=\\linewidth}",
                      "\\begin{tabular}{l l p{0.45\\linewidth} r}", "\\toprule",
                      "task & signal & background & merged at \\\\", "\\midrule", *lines,
                      "\\bottomrule", "\\end{tabular}", "\\end{adjustbox}", "\\end{table}", ""])


def build(root: pathlib.Path, ladder_path: pathlib.Path, vcb_path: pathlib.Path | None,
          anomaly_path: pathlib.Path | None, labels: dict, v2_levels: bool) -> dict:
    """{relative output path: text}; empty when the label map is not under root."""
    maps = root / "configs" / "labelmaps"
    if not (maps / "rung_label_maps.v1.csv").exists():
        return {}
    rows = read_map(maps / "rung_label_maps.v1.csv")
    if not all(r["class_name"].startswith("label_") for r in rows):
        return {}                                   # a synthetic map: no JetClass-II names to print
    parts = partitions(maps, rows)
    grid = root / "configs" / "arms" / "v2_grid.json"
    if grid.exists():
        check_grid_covered(grid, parts)
    out = {"tables/appendix_levels.tex": table_levels(rows, v2_levels),
           "tables/appendix_vocabulary.tex": table_vocabulary(rows, parts)}
    if (root / "experiments" / "EVAL" / "probe.py").exists():
        load = lambda p: json.loads(pathlib.Path(p).read_text()) if p and pathlib.Path(p).exists() else None
        out["tables/appendix_tasks.tex"] = table_tasks(root, rows, load(ladder_path), load(vcb_path),
                                                       load(anomaly_path), labels)
    return out
