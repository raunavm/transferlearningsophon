#!/usr/bin/env python3
"""X->bb and X->bs: the figure, the table and the macros of the paper's resonance-tagging section,
from experiments/XBS/fit_xbx.py's per-model files. The same code reads the stand-in fits (the dry
run) and the real samples, so nothing in the section is written by hand at ingestion.

Input: every <model>.json under --fits (any depth; fits_ihep/<sig>/<model>.json or the stand-in's
fits/<model>.json), merged by model over their "signals". --as maps a stand-in signal onto the
name it stands in for (xbb_jc2=xbb).

  figure   eps_S / sqrt(eps_B) against eps_B in the mass window, one panel per signal (Sophon,
           arXiv:2405.12972, Fig. 2, on our QCD background): Sophon's transfer classifier ("head",
           its seeds averaged per model) for every label set, mean over the pretraining runs with
           the range of the runs; the direct discriminant (Sophon Eq. 4) dashed where it can be
           formed; for X->bs the X->bq class's output (Sophon's control) dotted; the untrained
           network's linear probe.
  table    signal efficiency (%) at the fixed QCD efficiency eps_B* of each signal (A5: 1e-3 with
           at least 400,000 QCD jets in the window, else 1e-2), mean +- SD over the runs.
  macros   XbxMx, XbxWindowLo/Hi, and per signal the training jets per class, the repetition
           factor and the evaluation jets (signal and QCD) in the window.

    python3 experiments/XBS/xbx_report.py --fits /data/results/eval/xbx/fits --as xbb_jc2=xbb \\
        --out /data/results/xbx_dryrun/<commit>
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
SIGNALS = (("xbb", r"$X\to b\bar b$"), ("xbs", r"$X\to bs$"))
ROWS = ("L188", "L162", "R42_Q1", "R16_Q1", "L162_MASS", "R16_Q1_MASS", "RAND", "FLAV_F0", "FLAV_F1",
        "FLAV_F1R", "INIT")
CURVE_ROWS = ("L188", "L162", "R42_Q1", "R16_Q1", "RAND", "INIT")
WORD = {"xbb": "Xbb", "xbs": "Xbs"}


def _module(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load(fits: pathlib.Path, alias: dict | None = None) -> dict:
    """{model: {signal: readouts}} of every fit file under fits; a model's signals from several
    files are merged, and a signal given twice is refused."""
    out = {}
    for f in sorted(fits.rglob("*.json")):
        d = json.loads(f.read_text())
        if "signals" not in d or "model" not in d:
            continue
        sigs = out.setdefault(d["model"], {})
        for s, v in d["signals"].items():
            s = (alias or {}).get(s, s)
            if s in sigs:
                raise SystemExit(f"FATAL: {d['model']} {s} given twice ({f})")
            sigs[s] = v
    return out


def models_of(M, runs, row: str) -> list:
    if row == "RAND":
        return [m for a in M.RAND for m in runs[a]]
    if row == "INIT":
        return list(M.INIT_MODELS)
    return runs[row]


def readout(sig: dict, name: str) -> dict | None:
    """One readout's endpoints; the transfer classifier's seeds averaged (curve and working point)."""
    r = sig.get(name)
    if not r:
        return None
    if isinstance(r, dict):
        return r
    curves = [x["curve"] for x in r]
    return {"eps_s_at_star": float(np.mean([x["eps_s_at_star"] for x in r])),
            "eps_b_star": r[0]["eps_b_star"], "auc": float(np.mean([x["auc"] for x in r])),
            "curve": {"eps_b": curves[0]["eps_b"],
                      "eps_s_over_sqrt_eps_b": np.mean([c["eps_s_over_sqrt_eps_b"] for c in curves], axis=0).tolist()}}


def model_readout(fits: dict, model: str, signal: str) -> dict | None:
    """The readout a model is reported with: the transfer classifier, or the linear probe for a
    network without an output layer (the untrained one)."""
    sig = fits.get(model, {}).get(signal, {})
    return readout(sig, "head") or readout(sig, "linear")


def efficiencies(M, runs, fits: dict, signal: str, name: str | None = None) -> dict:
    """{row: [eps_S at eps_B* of each run]}; name picks a readout, default the reported one."""
    out = {}
    for row in ROWS:
        vals = []
        for m in models_of(M, runs, row):
            r = model_readout(fits, m, signal) if name is None else readout(fits.get(m, {}).get(signal, {}), name)
            if r is not None:
                vals.append(r["eps_s_at_star"])
        out[row] = vals
    return out


def table(M, MT, runs, fits: dict) -> str:
    """Signal efficiency (%) at eps_B* per configuration and signal; the direct discriminant for
    X->bb beside the transfer classifier where it can be formed."""
    have = [(s, t) for s, t in SIGNALS if any(s in v for v in fits.values())]
    if not have:
        return ""
    stars = {s: next(model_readout(fits, m, s)["eps_b_star"] for m in fits if model_readout(fits, m, s))
             for s, _ in have}
    cols = []
    for s, t in have:
        cols.append((s, None, f"{t}, transfer"))
        if s == "xbb":
            cols.append((s, "zero_shot_xbb", f"{t}, direct"))
    head = " & ".join(rf"{lab} ($\epsilon_B={stars[s]:g}$)" for s, _, lab in cols)
    lines = [r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule", rf"Pretraining & {head} \\", r"\midrule"]
    eff = {(s, n): efficiencies(M, runs, fits, s, n) for s, n, _ in cols}
    for row in ROWS:
        name = "random partitions (all five)" if row == "RAND" else M.LABEL[row]
        cells = []
        for s, n, _ in cols:
            v = eff[(s, n)][row]
            cells.append(M._cell(v, len(v), MT.fmt_pm, 100.0) if v else "---")
        if any(c != "---" for c in cells):
            lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def macros(em, MT, fits: dict, source: pathlib.Path) -> None:
    """The sample numbers the section quotes, identical across the models (refused otherwise)."""
    xbx = _module("experiments/XBS/xbx.py", "xbx")
    windows = {tuple(v["window"]) for sigs in fits.values() for v in sigs.values()}
    if len(windows) != 1:
        raise SystemExit(f"FATAL: the fits use {len(windows)} mass windows: {sorted(windows)}")
    lo, hi = windows.pop()
    mx = lo * xbx.SOPHON_MX / xbx.SOPHON_WINDOW[0]
    if not np.allclose(xbx.window_for(mx), (lo, hi)):
        raise SystemExit(f"FATAL: window {lo}-{hi} GeV is not Sophon's scaled by one resonance mass")
    em.macro("XbxMx", f"{mx:g}", source, "signals.*.window", "resonance mass: window / Sophon's at 200 GeV")
    em.macro("XbxWindowLo", f"{lo:g}", source, "signals.*.window[0]")
    em.macro("XbxWindowHi", f"{hi:g}", source, "signals.*.window[1]")
    for s, _ in SIGNALS:
        sig = [v[s] for v in fits.values() if s in v]
        if not sig:
            continue
        for key, mac, show in (("n_train_per_class", "NTrainPerClass", MT.fmt_int),
                               ("repetition", "Repetition", lambda x: f"{x:.1f}")):
            vals = {v[key] for v in sig}
            if len(vals) != 1:
                raise SystemExit(f"FATAL: {s} {key} differs across the models: {sorted(vals)}")
            em.macro(f"{WORD[s]}{mac}", show(vals.pop()), source, f"signals.{s}.{key}")
        ns = {(r["n_s"], r["n_b"]) for v in sig for r in [v["linear"]]}
        if len(ns) != 1:
            raise SystemExit(f"FATAL: {s} evaluation jets differ across the models: {sorted(ns)}")
        n_s, n_b = ns.pop()
        em.macro(f"{WORD[s]}NEvalSignal", MT.fmt_int(n_s), source, f"signals.{s}.linear.n_s")
        em.macro(f"{WORD[s]}NEvalQcd", MT.fmt_int(n_b), source, f"signals.{s}.linear.n_b")


def figure(M, style, runs, fits: dict, outdir: pathlib.Path) -> list:
    have = [(s, t) for s, t in SIGNALS if any(s in v for v in fits.values())]
    if not have:
        return []
    style.use_style()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(have), figsize=(0.8 * style.PANEL * len(have) + 0.4, 0.8 * style.PANEL),
                             squeeze=False)
    colour = {"RAND": style.RAND_COLOUR, "INIT": style.INIT_STYLE["color"]}
    level = {"L188": 188, "L162": 162, "R42_Q1": 43, "R16_Q1": 17}
    name = {"RAND": "Random partitions", "INIT": "Randomly initialized (linear)",
            **{k: f"{v} classes" for k, v in level.items()}}
    for ax, (s, title) in zip(axes[0], have):
        for row in CURVE_ROWS:
            c = colour.get(row) or style.LEVEL_COLOURS[level[row]]
            curves = [r["curve"] for m in models_of(M, runs, row) if (r := model_readout(fits, m, s))]
            if not curves:
                continue
            x = np.asarray(curves[0]["eps_b"])
            y = np.asarray([cv["eps_s_over_sqrt_eps_b"] for cv in curves])
            ax.plot(x, y.mean(axis=0), color=c, linewidth=2, linestyle=":" if row == "INIT" else "-",
                    label=name[row])
            if len(y) > 1:
                ax.fill_between(x, y.min(axis=0), y.max(axis=0), color=c, alpha=0.15, linewidth=0)
        extra = (("zero_shot_xbb", "--", "direct"),) if s == "xbb" else (("zero_shot_xbq", (0, (1, 1)), r"$X\to bq$ class"),)
        for key, ls, lab in extra:
            for row in ("L188", "L162"):
                curves = [r["curve"] for m in runs[row] if (r := readout(fits.get(m, {}).get(s, {}), key))]
                if curves:
                    x = np.asarray(curves[0]["eps_b"])
                    ax.plot(x, np.mean([cv["eps_s_over_sqrt_eps_b"] for cv in curves], axis=0), linestyle=ls,
                            color=style.LEVEL_COLOURS[level[row]], linewidth=1.5, label=f"{name[row]}, {lab}")
        ax.set_xscale("log")
        ax.set_title(title, fontsize="small", loc="left")
        ax.set_xlabel(r"QCD efficiency $\varepsilon_B$")
    axes[0, 0].set_ylabel(r"$\varepsilon_S/\sqrt{\varepsilon_B}$")
    axes[0, 0].legend(frameon=False, fontsize="xx-small")
    fig.tight_layout()
    return style.save(fig, "xbx", outdir)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fits", required=True, type=pathlib.Path)
    ap.add_argument("--as", dest="alias", nargs="*", default=[], help="stand-in=signal, e.g. xbb_jc2=xbb")
    ap.add_argument("--out", required=True, type=pathlib.Path)
    a = ap.parse_args(argv)
    M = _module("experiments/FIGS/mlst_results.py", "mlst_results")
    MT = _module("experiments/FIGS/make_tables.py", "make_tables")
    style = _module("experiments/FIGS/mlst_style.py", "mlst_style")
    fits = load(a.fits, dict(x.split("=", 1) for x in a.alias))
    if not fits:
        raise SystemExit(f"FATAL: no fit files under {a.fits}")
    runs = M._paper(REPO)
    a.out.mkdir(parents=True, exist_ok=True)
    paths = figure(M, style, runs, fits, a.out)
    (a.out / "xbx_table.tex").write_text(table(M, MT, runs, fits))
    em = MT.Emitter(pathlib.Path("/"))      # the fits live outside the repository here
    macros(em, MT, fits, sorted(a.fits.rglob("*.json"))[0].resolve())
    (a.out / "xbx_macros.tex").write_text("".join(f"\\newcommand{{\\{n}}}{{{b}}}  % {c}\n" for n, b, c in em.macros))
    print("\n".join(str(p) for p in paths + [a.out / "xbx_table.tex", a.out / "xbx_macros.tex"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
