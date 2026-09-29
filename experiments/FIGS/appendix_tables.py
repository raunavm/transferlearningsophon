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
PRETRAINED_V2 = {"R63_Q1", "R29_Q1"}                  # the 64- and 30-class rerun arms

_TOKENS = re.compile(r"tauh|taue|taum|[bcsqgemv]")
_TOKEN_TEX = {"tauh": r"\tau_h", "taue": r"\tau_e", "taum": r"\tau_\mu", "e": "e",
              "m": r"\mu", "v": r"\nu"}


def native_tex(name: str) -> str:
    """label_X_YY_bctauhv -> $X\\to YY\\to bc\\tau_h\\nu$; label_QCD_bbc -> QCD, $bbc$."""
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
    configs/labelmaps (any *label_map.v*.csv but the tree's own), each file's
    partitions in column order. The first grid's random partitions (v1) are
    numbered from 1; a later file's are headed by version and number (v2.3), or by
    their own name when it is not a numbered random partition (F0)."""
    out = []
    names = {r["class_name"] for r in rows}
    for f in sorted(maps_dir.glob("*label_map*.v*.csv")):
        if f.name.startswith("rung_label_maps"):
            continue
        version = re.search(r"\.v(\d+)\.csv$", f.name).group(1)
        prows = read_map(f)
        if {r["class_name"] for r in prows} != names:
            raise SystemExit(f"FATAL: {f} does not cover the native classes of the label map")
        cols = [c for c in prows[0] if re.fullmatch(r"(RAND|FLAV)\w*", c) and not c.endswith("_name")]
        for i, c in enumerate(cols, 1):
            m = re.fullmatch(r"RAND\d*_p(\d+)", c)
            head = (str(i) if version == "1" else f"v{version}.{m.group(1)}" if m
                    else c.split("_", 1)[-1])
            out.append((head, c, {r["class_name"]: r[c] for r in prows}))
    return out


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
    head = ("native class & " + " & ".join(str(sizes[r]) for r in SHOWN)
            + "".join(f" & {h}" for h, _, _ in parts) + " \\\\")
    body, last = [], None
    for r in sorted(rows, key=key):
        if last is not None and r["R16_Q1"] != last:
            body.append("\\addlinespace[2pt]")
        last = r["R16_Q1"]
        body.append(native_tex(r["class_name"]) + " & " + " & ".join(r[g] for g in SHOWN)
                    + "".join(f" & {p[r['class_name']]}" for _, _, p in parts) + " \\\\")
    first = [h for h, _, _ in parts if h.isdigit()]
    later = [h for h, _, _ in parts if not h.isdigit()]
    caption = ("Every native JetClass-II class and the group it belongs to at each coarser level "
               f"of the label tree (columns headed by the number of classes at that level: "
               f"{', '.join(str(sizes[r]) for r in SHOWN)}) and in each random partition. "
               "Group numbers are those of the released label maps; classes that share a number "
               "in a column share a class in that vocabulary. At the "
               f"{sizes['L162']}-class level every resonant class keeps its own label and the "
               f"QCD classes share one; at {sizes['R3_VIS']} classes the groups are the two-, "
               "three- and four-prong decays and QCD. Rows are ordered by the "
               f"{sizes['R16_Q1']}-class group (Table~\\ref{{tab:app-levels}}), with a gap between "
               "groups. $q$ is a $u$ or $d$ quark; $\\tau_h$ a hadronic and $\\tau_e$, $\\tau_\\mu$ "
               "leptonic $\\tau$ decays.")
    if first:
        caption += (f" Random partitions {', '.join(first)} are those of Sec.~\\ref{{sec:controls}}")
        caption += (f"; {', '.join(later)} those of the rerun." if later else ".")
    ncol = len(SHOWN) + len(parts)
    return _longtable("l " + "r" * ncol, head, body, caption, "tab:app-vocabulary")


def table_levels(rows: list[dict], v2_levels: bool) -> str:
    """The levels of the tree, and the coarsest pretrained vocabulary in words."""
    sizes = {rung: len({r[rung] for r in rows}) for rung in RUNGS}
    rule = {"L188": "native classes",
            "L162": "one QCD class; every resonant class kept",
            "R63_Q1": "each decay topology by its numbers of $b$ and $c$ quarks",
            "R42_Q1": "each decay topology by whether it holds a $b$ quark, else a $c$, else neither",
            "R29_Q1": "each decay topology by whether it holds a $b$ or $c$ quark",
            "R16_Q1": "decay topology: prongs, and quarks or gluons and leptons, without flavour",
            "R3_VIS": "two-, three- or four-prong decay, or QCD",
            "R1_Q1": "resonance decay or QCD"}
    pre = lambda r: ("yes" if r in PRETRAINED else
                     ("yes (rerun)" if v2_levels else "\\pending{rerun}") if r in PRETRAINED_V2 else "no")
    body = [f"{sizes[r]} & {rule[r]} & {pre(r)} \\\\" for r in RUNGS]
    body.append("\\midrule")
    groups = collections.OrderedDict()
    for r in rows:
        groups.setdefault((r["R16_Q1"], r["R16_Q1_name"]), []).append(r)
    body.append(f"\\multicolumn{{3}}{{@{{}}l}}{{\\itshape the {sizes['R16_Q1']} classes of the "
                f"{sizes['R16_Q1']}-class vocabulary}} \\\\")
    for (gid, name), members in groups.items():
        body.append(f"{gid} & {group_tex(name)} & {len(members)} native \\\\")
    caption = ("The levels of the label tree, finest first: the number of classes, what each level "
               "labels, and whether models were pretrained on it. Below, the groups of the "
               f"{sizes['R16_Q1']}-class vocabulary, by the group number used in "
               "Table~\\ref{tab:app-vocabulary}, with the number of native classes each merges. "
               "$\\ell$ is an electron or a muon; $\\tau_\\ell$ a leptonic $\\tau$ decay.")
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
    out = {"tables/appendix_levels.tex": table_levels(rows, v2_levels),
           "tables/appendix_vocabulary.tex": table_vocabulary(rows, partitions(maps, rows))}
    if (root / "experiments" / "EVAL" / "probe.py").exists():
        load = lambda p: json.loads(pathlib.Path(p).read_text()) if p and pathlib.Path(p).exists() else None
        out["tables/appendix_tasks.tex"] = table_tasks(root, rows, load(ladder_path), load(vcb_path),
                                                       load(anomaly_path), labels)
    return out
