"""The revision checks of 2026-11-26 (external audit of draft 8f73f2d, V1-V3, A2, item 10), read through the v2 build
(experiments/FIGS/mlst_v2_tables.py calls macros(); every number here is a macro of numbers.tex).

  CS-15     the two fine-tuning checks on the 188-class run 1 (pretrained output layer kept; new output layer at the
            backbone's rate) against the same run's fine-tuned cell and linear probe (bootstrap/l3_fixed/cs15_*)
  scratch   the resonance benchmark's classifier trained from scratch, run by run, with its training curves
            (xbx_scratch_training/curves.json) and the 17-class transfer head minus it at 1% and 0.1% background
  mass      the frozen 17 + mass regression runs and the pretraining runs' own validation accuracy (compute/)
  recall    mean change of per-class recall at 43 and 17 classes by the kind of merge a class undergoes
  labels    a classifier that knows only a jet's group and guesses uniformly within it
"""
from __future__ import annotations

import csv
import json
import math
import pathlib

import numpy as np

CS15 = "bootstrap/l3_fixed/cs15_jc2bal_N1000000.json"
CURVES = "xbx_scratch_training/curves.json"
SIGMA = 3.0


def _sd(v):
    return float(np.std(v, ddof=1)) if len(v) > 1 else None


def cs15(data: pathlib.Path, ft_vals: list, lp_vals: list) -> tuple[dict, list]:
    """Macros and report lines of the two CS-15 checks. ft_vals / lp_vals: the three 188-class runs' fine-tuned and
    probe accuracies as Tables 3 and 2 print them (the build stops unless the file holds the same values). A check is
    one run, so its run SD is taken as the fine-tuned 188-class row's (as is run 1's when it is one side alone);
    sigma^2 = the two sides' run SD^2 + the paired test-jet bootstrap SD^2 (SC-1)."""
    doc = json.loads((data / CS15).read_text())
    m = {k: v["accuracy"]["value"] for k, v in doc["models"].items()}
    ft = [m[f"ft-mtx-l188-s{k}"] for k in (1, 2, 3)]
    lp = [m[f"lp-mtx-l188-s{k}"] for k in (1, 2, 3)]
    # each run's value in the file within a quarter of its bootstrap SD of the one the table prints (the rule of the
    # build's "file" rows; the probes' stored probabilities are float16, so their argmax can differ on a few jets)
    for k, (what, want) in enumerate((("ft", ft_vals), ("lp", lp_vals))):
        for i, w in enumerate(want, 1):
            rec = doc["models"][f"{what}-mtx-l188-s{i}"]["accuracy"]
            if abs(rec["value"] - w) > 0.25 * rec["sd"]:
                raise SystemExit(f"FATAL: {CS15}: {what} run {i} is {rec['value']}, the table prints {w}")
    sft, slp = _sd(ft), _sd(lp)
    out, lines = {}, []
    for k, w in (("headkept", "HeadKept"), ("head1x", "HeadRate")):
        a = m[f"cs15-{k}"]
        out[f"VCs{w}Acc"] = f"{100 * a:.2f}"
        for ref, rw, sa, sb in ((f"{k}-ft1", "Ft", sft, sft), (f"{k}-lp1", "Lp", sft, slp)):
            e = doc["diffs"][ref]["accuracy"]
            sig = math.sqrt(sa ** 2 + sb ** 2 + e["boot_sd"] ** 2)
            z = e["delta"] / sig
            out[f"VDCs{w}{rw}"] = f"{100 * e['delta']:+.2f}"
            out[f"VSigCs{w}{rw}"] = f"{100 * sig:.2f}"
            out[f"VZCs{w}{rw}"] = f"{z:+.1f}"
            lines.append(f"{k} - {'run-1 fine-tuned' if rw == 'Ft' else 'run-1 probe'}: delta {100 * e['delta']:+.3f} pts, "
                         f"boot {100 * e['boot_sd']:.3f}, sigma {100 * sig:.3f}, delta/sigma {z:+.2f}")
    e = doc["diffs"]["ft1-lp1"]["accuracy"]
    sig = math.sqrt(sft ** 2 + slp ** 2 + e["boot_sd"] ** 2)
    out.update({"VCsFtOneAcc": f"{100 * ft[0]:.2f}", "VCsLpOneAcc": f"{100 * lp[0]:.2f}",
                "VDCsFtOneLp": f"{100 * e['delta']:+.2f}", "VZCsFtOneLp": f"{e['delta'] / sig:+.1f}"})
    lines.append(f"run-1 fine-tuned - run-1 probe: delta {100 * e['delta']:+.3f} pts, sigma {100 * sig:.3f}, "
                 f"delta/sigma {e['delta'] / sig:+.2f}; run SDs fine-tuned {100 * sft:.3f}, probe {100 * slp:.3f} pts")
    lines.insert(0, f"accuracy: head kept {100 * m['cs15-headkept']:.3f}%, head rate 1e-4 {100 * m['cs15-head1x']:.3f}%, "
                    f"run-1 fine-tuned {100 * ft[0]:.3f}%, run-1 probe {100 * lp[0]:.3f}%")
    return out, lines


def eps_at(rec: dict, eps_b: float) -> tuple[float, float]:
    """(signal efficiency at the stored ROC point nearest eps_b, its analytic variance): binomial on the signal jets
    plus the slope^2 term of the background jets through the cut, as mlst_results.table_tests' transfer-head rows."""
    c = rec["curve"]
    eb, s = np.array(c["eps_b"]), np.array(c["eps_s"])
    i = int(np.argmin(abs(np.log(eb) - math.log(eps_b))))
    if abs(eb[i] / eps_b - 1) > 0.01:
        raise SystemExit(f"FATAL: no stored ROC point at a background efficiency of {eps_b} (nearest {eb[i]})")
    slope = (s[i + 1] - s[i - 1]) / (eb[i + 1] - eb[i - 1])
    e = float(s[i])
    return e, e * (1 - e) / rec["n_s"] + slope ** 2 * eb[i] * (1 - eb[i]) / rec["n_b"]


def scratch(data: pathlib.Path, XR, xfits: dict, scr: dict, coarse_runs: list) -> tuple[dict, list]:
    """The from-scratch resonance classifier run by run (seed, signal efficiency at 1% and 0.1% background, AUC,
    best epoch, its training record) and the 17-class transfer head minus it, sigma^2 = the two sides' run SD^2 plus
    their analytic test-sample terms (SC-2b's form, sides added)."""
    curves = json.loads((data / CURVES).read_text())["runs"]
    out, lines = {}, []
    for sig, w in (("xbb", "Xbb"), ("xbs", "Xbs")):
        recs = scr.get(sig) or []
        if len(recs) != 3:
            raise SystemExit(f"FATAL: {sig}: {len(recs)} from-scratch runs")
        for k, r in enumerate(recs, 1):
            cv = curves[f"{sig}/s{k}"]
            if int(cv["manifest"]["ft_seed"]) != k or cv["first_nonfinite_epoch"] is not None:
                raise SystemExit(f"FATAL: {sig} s{k}: seed {cv['manifest']['ft_seed']} or a non-finite loss")
            best = cv["best_epoch"]["epoch"]
            n = cv["best_epoch"]["n_epochs"]
            # the failed-run rules of the one fine-tuning rule: no best check first or in the last 10% (FT-3), no
            # non-finite loss (FT-10); a run that breaks one is named here, never averaged in silently
            fail = best < 1 or best >= 0.9 * n
            out[f"V{w}ScratchS{k}"] = f"{100 * eps_at(r, 0.01)[0]:.1f}"
            out[f"V{w}ScratchS{k}Milli"] = f"{100 * eps_at(r, 0.001)[0]:.1f}"
            lines.append(f"{sig} from scratch s{k} (seed {k}): eps_S {100 * eps_at(r, 0.01)[0]:.2f}% at 1%, "
                         f"{100 * eps_at(r, 0.001)[0]:.2f}% at 0.1%; AUC {r['auc']:.4f}; best epoch {best} of {n}, "
                         f"best validation {cv['best_epoch']['metric']:.4f}, last {cv['val_metric'][-1]:.4f}; final "
                         f"train loss {cv['train_loss'][-1]:.4f}; GPU {cv['manifest']['gpu_device_name']}; "
                         f"{'FAILS the edge rule' if fail else 'passes the edge and divergence rules'}")
        heads = [XR.readout(xfits.get(m, {}).get(sig, {}), "head") for m in coarse_runs]
        for eb, ew in ((0.01, ""), (0.001, "Milli")):
            a = [eps_at(h, eb) for h in heads]
            b = [eps_at(r, eb) for r in recs]
            va, vb = [x for x, _ in a], [x for x, _ in b]
            boot = math.hypot(math.sqrt(float(np.mean([v for _, v in a]))), math.sqrt(float(np.mean([v for _, v in b]))))
            d = float(np.mean(va) - np.mean(vb))
            s = math.sqrt(_sd(va) ** 2 + _sd(vb) ** 2 + boot ** 2)
            out[f"VD{w}CoarseScratch{ew}"] = f"{100 * d:+.1f}"
            out[f"VSig{w}CoarseScratch{ew}"] = f"{100 * s:.1f}"
            out[f"VZ{w}CoarseScratch{ew}"] = f"{d / s:+.1f}"
            out[f"V{w}Scratch{ew}Pm"] = f"{100 * np.mean(vb):.2f} $\\pm$ {100 * _sd(vb):.2f}"
            lines.append(f"{sig} at {100 * eb:g}% background: 17 classes {100 * np.mean(va):.2f} +- {100 * _sd(va):.2f}, "
                         f"from scratch {100 * np.mean(vb):.2f} +- {100 * _sd(vb):.2f}; 17 - scratch {100 * d:+.2f} pts, "
                         f"sigma {100 * s:.2f} (run SDs {100 * _sd(va):.2f}, {100 * _sd(vb):.2f}; test term {100 * boot:.2f}), "
                         f"delta/sigma {d / s:+.2f}")
    return out, lines


def pretraining_val(data: pathlib.Path, run: str) -> tuple[float, float, int]:
    """(validation accuracy of the pretraining run at the epoch its checkpoint was taken (best70), at its last epoch,
    that epoch) from compute/<run>.json and ckpt_identity."""
    ep = json.loads((data / "compute" / f"{run}.json").read_text())["epochs"]
    ck = json.loads((data / "ckpt_identity/identity.json").read_text())["runs"][run]["best70"]
    e = int(ck.removeprefix("net_epoch-").removesuffix("_state.pt"))
    at = {x["epoch"]: x["val_acc"] for x in ep}
    return at[e], ep[-1]["val_acc"], e


def mass(data: pathlib.Path, lp_runs: dict) -> tuple[dict, list]:
    """The frozen JetClass-II accuracy of each run of the 17-class sets with and without the mass regression, with the
    pretraining run's own validation accuracy (on its 17 classes) at its checkpoint."""
    out, lines = {}, []
    for arm, w, runs in (("R16_Q1_MASS", "CoarseMass", lp_runs["R16_Q1_MASS"]), ("R16_Q1", "Coarse", lp_runs["R16_Q1"]),
                         ("L162_MASS", "ResMass", lp_runs["L162_MASS"]), ("L162", "Res", lp_runs["L162"])):
        for k, (m, v) in enumerate(runs, 1):
            pv, last, e = pretraining_val(data, m)
            out[f"VLpJcTwo{w}S{k}"] = f"{100 * v:.1f}"
            out[f"VPreVal{w}S{k}"] = f"{100 * pv:.2f}"
            lines.append(f"{arm} {m}: frozen JetClass-II accuracy {100 * v:.2f}%; pretraining validation accuracy "
                         f"{100 * pv:.3f}% at epoch {e} (the checkpoint), {100 * last:.3f}% at the last epoch")
    return out, lines


def recall_by_merge(R, root: pathlib.Path, docs: dict, runs: dict) -> tuple[dict, list]:
    """Mean over classes of the change of per-class recall (the arm's mean over runs minus 188 classes'), frozen
    JetClass-II, at 43 and 17 classes, by what the label set merges a class with: QCD classes; resonant classes merged
    with a decay that differs only in quark flavor, only in lepton flavor, with both kinds, or with neither."""
    q, lep = R._orbits(root)
    groups = R._groups(root)
    names = {int(r["jet_label"]): r["class_name"] for r in csv.DictReader((root / "configs/labelmaps/rung_label_maps.v1.csv").open())}
    ref = R.per_class_recall(docs, runs, "L188", "lp")
    out, lines = {}, []
    for arm, w in (("R42_Q1", "Mid"), ("R16_Q1", "Coarse")):
        g = groups[arm]
        rec = R.per_class_recall(docs, runs, arm, "lp")
        if len(rec) != 188 or len(ref) != 188 or any(len(v) != len(runs[arm]) for v in rec.values()):
            raise SystemExit(f"FATAL: per-class recall of {arm}: {len(rec)} classes")

        def merged(c, key_of):
            n = names[c]
            return n in key_of and any(b != n and key_of[b] == key_of[n] and g[b] == g[n] for b in key_of)
        kinds = {"Qcd": [], "Quark": [], "Lepton": [], "Both": [], "Neither": []}
        for c in rec:
            d = 100 * (float(np.mean(rec[c])) - float(np.mean(ref[c])))
            if names[c].startswith("label_QCD"):
                kinds["Qcd"].append(d)
                continue
            mq, ml = merged(c, q), merged(c, lep)
            kinds["Both" if mq and ml else "Quark" if mq else "Lepton" if ml else "Neither"].append(d)
        for k, v in kinds.items():
            out[f"VRecall{w}{k}"] = f"{float(np.mean(v)):.1f}" if v else "--"
            out[f"VRecall{w}{k}N"] = str(len(v))
        lines.append(f"{arm}: " + "; ".join(f"{k} {len(v)} classes, mean recall change {float(np.mean(v)):+.2f} pts"
                                            if v else f"{k} 0 classes" for k, v in kinds.items()))
    return out, lines


def label_only(root: pathlib.Path, maps: dict, counts: list) -> tuple[dict, list]:
    """The accuracy on the class-balanced JetClass-II test jets (counts: jets per class, by jet_label) of a classifier
    that knows each jet's group and picks one of the group's classes uniformly at random: sum over classes of
    n_c / |group(c)|, over the jets."""
    names = {int(r["jet_label"]): r["class_name"].removeprefix("label_")
             for r in csv.DictReader((root / "configs/labelmaps/rung_label_maps.v1.csv").open())}
    n = np.asarray(counts, dtype=float)
    out, lines = {}, []
    sets = [("L162", "Res"), ("R42_Q1", "Mid"), ("R16_Q1", "Coarse")] + [(c, f"RandP{c[-1]}") for c in sorted(maps)
                                                                       if c.startswith("RAND2_p")]
    rand = []
    for arm, w in sets:
        g = maps[arm]
        size = {}
        for c in g.values():
            size[c] = size.get(c, 0) + 1
        acc = float(sum(n[k] / size[g[names[k]]] for k in names) / n.sum())
        out[f"VLabelOnly{w}"] = f"{100 * acc:.1f}"
        if arm.startswith("RAND2_p"):
            rand.append(acc)
        lines.append(f"{arm}: {len(size)} groups, label-only accuracy {100 * acc:.2f}% (groups/188 = {100 * len(size) / 188:.2f}%)")
    out["VLabelOnlyRand"] = f"{100 * float(np.mean(rand)):.1f}"
    lines.append(f"random groupings: mean {100 * float(np.mean(rand)):.2f}%, range {100 * min(rand):.2f}-{100 * max(rand):.2f}%")
    return out, lines


def macros(got: dict, maps: dict, runs: dict, lp_vals, ft_vals, root: pathlib.Path) -> tuple[dict, list]:
    """Every macro of the revision checks and the lines of their report (paper/journal/v2/tables/rev35.txt).
    lp_vals / ft_vals: f(column, arm) -> the run values Tables 2 and 3 print."""
    R = got["R"]
    data = root / "experiments/FIGS/data/v2"
    out, lines = {}, []
    o, ln = cs15(data, ft_vals("JetClass-II accuracy", "L188"), lp_vals("JetClass-II accuracy", "L188"))
    out.update(o); lines += ["## CS-15"] + ln
    xbx = got["summary_kw"].get("xbx") or {}
    o, ln = scratch(data, xbx["XR"], xbx["fits"], xbx.get("scr") or {}, runs["R16_Q1"])
    out.update(o); lines += ["## resonance benchmark from scratch"] + ln
    lp_runs = {a: list(zip(runs[a], lp_vals("JetClass-II accuracy", a))) for a in ("R16_Q1_MASS", "R16_Q1", "L162_MASS", "L162")}
    o, ln = mass(data, lp_runs)
    out.update(o); lines += ["## mass regression, frozen"] + ln
    docs = {"lp": json.loads((data / "perclass_recall/probe_jc2.json").read_text())}
    o, ln = recall_by_merge(R, root, docs, runs)
    out.update(o); lines += ["## per-class recall by merge"] + ln
    o, ln = label_only(root, maps, docs["lp"]["class_counts"])
    out.update(o); lines += ["## label-only reference"] + ln
    return out, lines
