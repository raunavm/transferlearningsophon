#!/usr/bin/env python3
"""The label-set palette of figure round 4 (docs/figstyle/ROUND4_BRIEF.md, Part 2), checked and chosen.

For each candidate family (Tol bright, Tol muted, Okabe-Ito without orange and black, in the brief's order) every
choice of four hues is scored: pairwise CIEDE2000 in normal vision (rule: >= 25), and under deuteranopia and
protanopia simulated with Machado, Oliveira and Fernandes (2009) at severity 1 (rule: >= DE_CVD); the lightness L*
(rule: falling from 17 to 188 classes, at least DL_STEP per step, so the hierarchy reads in greyscale). No colour
may be orange, black or a grey (chroma below CHROMA_MIN). The assignment puts the darkest colour on 188 classes.
A family passes when one choice meets every rule; otherwise the luminance rule is relaxed first (PI, 2026-11-07),
and the report says what the relaxation buys.

    python3 -I experiments/FIGS/figstyle/palette_check.py            # the report
    python3 experiments/FIGS/figstyle/palette_check.py --networks    # every network colour of style_v5 (L6)
"""
from __future__ import annotations

import itertools
import json
import math
import sys

import numpy as np

FAMILIES = {
    "Tol bright": {"blue": "#4477AA", "cyan": "#66CCEE", "green": "#228833", "yellow": "#CCBB44",
                   "red": "#EE6677", "purple": "#AA3377"},
    "Tol muted": {"indigo": "#332288", "cyan": "#88CCEE", "teal": "#44AA99", "green": "#117733",
                  "olive": "#999933", "sand": "#DDCC77", "rose": "#CC6677", "wine": "#882255", "purple": "#AA4499"},
    "Okabe-Ito": {"sky blue": "#56B4E9", "bluish green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2",
                  "vermillion": "#D55E00", "reddish purple": "#CC79A7"},
}
BRIEF_BRIGHT = ("blue", "cyan", "green", "purple")      # the brief's first candidate
LEVELS = ("188", "162", "43", "17")                     # darkest first
DE_MIN, DE_CVD, DL_STEP, CHROMA_MIN = 25.0, 10.0, 5.0, 20.0
ORANGE_HUE = (40.0, 75.0)                               # CIELab hue angle of orange (degrees), with chroma

MACHADO = {
    "deuteranopia": np.array([[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413],
                              [-0.011820, 0.042940, 0.968881]]),
    "protanopia": np.array([[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216],
                            [-0.003882, -0.048116, 1.051998]]),
}


def rgb(hexcode: str) -> np.ndarray:
    return np.array([int(hexcode[i:i + 2], 16) / 255 for i in (1, 3, 5)])


def to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def simulate(c: np.ndarray, kind: str) -> np.ndarray:
    """sRGB in [0, 1] (any shape ending in 3) seen with kind (Machado 2009, severity 1)."""
    return to_srgb(to_linear(c) @ MACHADO[kind].T)


def lab(c: np.ndarray) -> np.ndarray:
    """sRGB in [0, 1] to CIELab (D65)."""
    m = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = to_linear(c) @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def ciede2000(l1: np.ndarray, l2: np.ndarray) -> float:
    """CIEDE2000 (Sharma, Wu and Dalal 2005) between two CIELab colours."""
    L1, a1, b1 = l1
    L2, a2, b2 = l2
    cbar = (math.hypot(a1, b1) + math.hypot(a2, b2)) / 2
    g = 0.5 * (1 - math.sqrt(cbar ** 7 / (cbar ** 7 + 25 ** 7)))
    a1p, a2p = (1 + g) * a1, (1 + g) * a2
    c1p, c2p = math.hypot(a1p, b1), math.hypot(a2p, b2)
    h1p = math.degrees(math.atan2(b1, a1p)) % 360
    h2p = math.degrees(math.atan2(b2, a2p)) % 360
    dlp, dcp = L2 - L1, c2p - c1p
    if c1p * c2p == 0:
        dhp = 0.0
    elif abs(h2p - h1p) <= 180:
        dhp = h2p - h1p
    else:
        dhp = h2p - h1p - 360 if h2p > h1p else h2p - h1p + 360
    dHp = 2 * math.sqrt(c1p * c2p) * math.sin(math.radians(dhp / 2))
    lbp, cbp = (L1 + L2) / 2, (c1p + c2p) / 2
    if c1p * c2p == 0:
        hbp = h1p + h2p
    elif abs(h1p - h2p) <= 180:
        hbp = (h1p + h2p) / 2
    else:
        hbp = (h1p + h2p + 360) / 2 if h1p + h2p < 360 else (h1p + h2p - 360) / 2
    t = (1 - 0.17 * math.cos(math.radians(hbp - 30)) + 0.24 * math.cos(math.radians(2 * hbp))
         + 0.32 * math.cos(math.radians(3 * hbp + 6)) - 0.20 * math.cos(math.radians(4 * hbp - 63)))
    dtheta = 30 * math.exp(-(((hbp - 275) / 25) ** 2))
    rc = 2 * math.sqrt(cbp ** 7 / (cbp ** 7 + 25 ** 7))
    sl = 1 + 0.015 * (lbp - 50) ** 2 / math.sqrt(20 + (lbp - 50) ** 2)
    sc, sh = 1 + 0.045 * cbp, 1 + 0.015 * cbp * t
    rt = -math.sin(math.radians(2 * dtheta)) * rc
    return math.sqrt((dlp / sl) ** 2 + (dcp / sc) ** 2 + (dHp / sh) ** 2 + rt * (dcp / sc) * (dHp / sh))


def excluded(hexcode: str) -> str | None:
    """Why a colour may not be a label set's: orange, black or grey; None when allowed."""
    L, a, b = lab(rgb(hexcode))
    chroma, hue = math.hypot(a, b), math.degrees(math.atan2(b, a)) % 360
    if L < 15:
        return "black"
    if chroma < CHROMA_MIN:
        return "grey"
    if ORANGE_HUE[0] <= hue <= ORANGE_HUE[1] and chroma > 40:
        return "orange"
    return None


def score(colours: list[str]) -> dict:
    """The rules for one ordered choice (188, 162, 43, 17)."""
    c = np.array([rgb(h) for h in colours])
    out = {"colours": dict(zip(LEVELS, colours)), "L": [round(float(x), 1) for x in lab(c)[:, 0]]}
    for kind in ("normal", *MACHADO):
        seen = c if kind == "normal" else simulate(c, kind)
        L = lab(seen)
        out[f"min_de_{kind}"] = round(min(ciede2000(L[i], L[j]) for i, j in itertools.combinations(range(4), 2)), 1)
    steps = np.diff(out["L"])                 # 188 -> 162 -> 43 -> 17 must grow lighter
    out["min_L_step"] = round(float(steps.min()), 1)
    out["luminance_ok"] = bool(steps.min() >= DL_STEP)
    out["de_ok"] = out["min_de_normal"] >= DE_MIN
    out["cvd_ok"] = min(out["min_de_deuteranopia"], out["min_de_protanopia"]) >= DE_CVD
    out["passes"] = out["de_ok"] and out["cvd_ok"] and out["luminance_ok"]
    return out


def best(family: dict, need_luminance: bool) -> dict | None:
    """The choice of four allowed colours, darkest on 188, that meets the rules (the luminance rule only when
    need_luminance) with the largest worst-case separation; None when none does."""
    allowed = [h for h in family.values() if not excluded(h)]
    found = []
    for four in itertools.combinations(allowed, 4):
        order = sorted(four, key=lambda h: lab(rgb(h))[0])
        s = score(order)
        if s["de_ok"] and s["cvd_ok"] and (s["luminance_ok"] or not need_luminance):
            found.append(s)
    if not found:
        return None
    return max(found, key=lambda s: (min(s["min_de_normal"], s["min_de_deuteranopia"], s["min_de_protanopia"]),
                                     s["min_L_step"]))


def report() -> dict:
    names = {h: f"{fam} {n}" for fam, cols in FAMILIES.items() for n, h in cols.items()}
    bright = [FAMILIES["Tol bright"][n] for n in BRIEF_BRIGHT]
    out = {"rules": {"de_min": DE_MIN, "de_cvd": DE_CVD, "dl_step": DL_STEP},
           "excluded": {names[h]: why for h in names if (why := excluded(h))},
           "brief_bright_as_given": score(bright),
           "brief_bright_by_lightness": score(sorted(bright, key=lambda h: lab(rgb(h))[0]))}
    for fam, cols in FAMILIES.items():
        out[fam] = {"all_rules": best(cols, True), "luminance_relaxed": best(cols, False)}
    for k, v in out.items():
        if isinstance(v, dict):
            for s in ([v] if "colours" in v else [x for x in v.values() if isinstance(x, dict) and "colours" in x]):
                s["names"] = {lv: names.get(h, h) for lv, h in s["colours"].items()}
    return out


# ------------------------------------------------------------------ the PI's palette (2026-11-07)
PI_LEVELS = {"188": "#332288", "162": "#AA4499", "43": "#999933", "17": "#44AA99"}     # Tol muted, as the PI chose
PI_CONTROLS = {"no flavor boundary": "#CC6677", "b-quark boundary": "#117733", "random flavor boundary": "#DDCC77"}
SPARES = {"cyan": "#88CCEE", "wine": "#882255"}         # the PI's swaps for a control that collides with a label set


def separation(a: str, b: str) -> dict:
    """CIEDE2000 between two colours in normal vision and under the two simulations."""
    c = np.array([rgb(a), rgb(b)])
    return {kind: round(ciede2000(*lab(c if kind == "normal" else simulate(c, kind))), 1) for kind in ("normal", *MACHADO)}


def collides(a: str, b: str) -> bool:
    """The label-set rule applied to a control against a label set: CIEDE2000 below DE_MIN, or below DE_CVD simulated."""
    d = separation(a, b)
    return d["normal"] < DE_MIN or min(d["deuteranopia"], d["protanopia"]) < DE_CVD


def pi_palette() -> dict:
    """The PI's controls checked against the label sets; each colliding control is swapped for the spare (cyan or wine)
    that leaves the smallest control-to-label-set separation largest. Returns the chosen controls, the swaps, and every
    pairwise separation among the seven colours."""
    worst = lambda cols: min(separation(c, l)["normal"] for c in cols for l in PI_LEVELS.values())
    controls, swaps = dict(PI_CONTROLS), []
    for name in sorted(PI_CONTROLS, key=lambda n: worst([PI_CONTROLS[n]])):
        if not any(collides(controls[name], l) for l in PI_LEVELS.values()):
            continue
        free = [h for h in SPARES.values() if h not in controls.values()]
        best = max(free + [controls[name]], key=lambda h: worst([*(c for n, c in controls.items() if n != name), h]))
        if best != controls[name]:
            swaps.append({"control": name, "from": controls[name], "to": best})
            controls[name] = best
    names = {**{f"{k} classes": v for k, v in PI_LEVELS.items()}, **controls}
    pairs = {f"{a} / {b}": separation(names[a], names[b]) for a, b in itertools.combinations(names, 2)}
    left = [{"control": n, "label set": f"{k} classes", **separation(h, l)} for n, h in controls.items()
            for k, l in PI_LEVELS.items() if collides(h, l)]
    return {"levels": PI_LEVELS, "controls": controls, "swaps": swaps, "pairs": pairs, "still_below_rule": left}


# ------------------------------------------------------------------ every network colour (PI 2026-11-13, L6)
# the networks whose curves can share one panel, by figure family (style_v5.SERIES keys): the label sets with the mass
# networks and the mode's references, and the 17-class partitions with 188 and 17 classes as references. The headline
# dot plots show every network but name each row, so they are not a group.
GROUPS = {
    "label sets, fine-tuned": ("L188", "L162", "L162_MASS", "R42_Q1", "R16_Q1", "R16_Q1_MASS", "KIN", "SCRATCH"),
    "label sets, linear probes": ("L188", "L162", "L162_MASS", "R42_Q1", "R16_Q1", "R16_Q1_MASS", "KIN", "SOPHON",
                                  "INIT_BN"),
    "partitions, fine-tuned": ("L188", "R16_Q1", "FLAV_F0", "FLAV_F1", "FLAV_F1R", "RAND", "SCRATCH"),
    "partitions, linear probes": ("L188", "R16_Q1", "FLAV_F0", "FLAV_F1", "FLAV_F1R", "RAND", "INIT_BN"),
    # the headline dot plots (F2, F2a, F2b): every network of a mode in one panel, each row named beside it
    "headline, linear probes": ("L188", "L162", "L162_MASS", "R42_Q1", "R16_Q1", "R16_Q1_MASS", "FLAV_F0", "FLAV_F1",
                                "FLAV_F1R", "RAND", "INIT_BN", "SOPHON"),
    "headline, fine-tuned": ("L188", "L162", "L162_MASS", "R42_Q1", "R16_Q1", "R16_Q1_MASS", "FLAV_F0", "FLAV_F1",
                             "FLAV_F1R", "RAND", "SCRATCH"),
}
REFERENCE_BLACK = "#000000"     # the soft-drop mass and the dotted reference lines share panels with the networks
SEPARABLE = DE_CVD              # a pair is told apart when CIEDE2000 reaches this in every view


def greyscale_l(hexcode: str) -> float:
    """L* of the colour printed in greyscale (the luminance, as gallery_palette's greyscale view)."""
    g = to_srgb(to_linear(rgb(hexcode)) @ np.array([0.2126, 0.7152, 0.0722]))
    return float(lab(np.array([g, g, g]))[0])


def networks() -> dict:
    """Every pair of network colours (style_v5.SERIES) that can share a panel (GROUPS), and each network against black:
    CIEDE2000 in normal vision and under the two simulations, and the L* difference in greyscale; the pairs below
    SEPARABLE in a view; and CLASS10 (the classes, families and UMAP groups) against every network colour."""
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
    import style_v5 as S
    pairs, low = {}, []
    for group, keys in GROUPS.items():
        for a, b in itertools.combinations(keys + ("black",), 2):
            ca, cb = (S.SERIES.get(k, REFERENCE_BLACK) for k in (a, b))
            d = {**separation(ca, cb), "greyscale_dL": round(abs(greyscale_l(ca) - greyscale_l(cb)), 1)}
            pairs[f"{a} / {b}"] = d
            if min(d["normal"], d["deuteranopia"], d["protanopia"]) < SEPARABLE:
                low.append({"pair": f"{a} / {b}", "group": group, **d})
    low = list({x["pair"]: x for x in low}.values())
    classes = {c: round(min(separation(c, h)["normal"] for h in S.SERIES.values()), 1) for c in S.CLASS10}
    return {"colours": dict(S.SERIES), "pairs": pairs, "below_separable_in_a_view": low,
            "class10_reuses_a_network_colour": sorted(set(S.CLASS10) & set(S.SERIES.values())),
            "class10_least_CIEDE2000_to_a_network": classes}


if __name__ == "__main__":
    json.dump(networks() if "--networks" in sys.argv else pi_palette() if "--pi" in sys.argv else report(), sys.stdout,
              indent=1)
    print()
