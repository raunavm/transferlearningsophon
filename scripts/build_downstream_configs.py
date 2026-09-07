#!/usr/bin/env python3
"""Emit the downstream data configs and their zero-fill CONTROLS from one table.

THE PROBLEM THIS SOLVES
-----------------------
Two of the paper's benchmark rows do not carry all 17 of Sophon's per-particle
features (`docs/PRD_PLAN.md` §4.5, RED item 16d):

    Top Quark Tagging Reference   kinematics only -- missing 10 of 17
    EnergyFlow q/g                no displacement -- missing 4 of 17

The decision is zero-fill, the convention of every 2025-26 paper, *conditional on
a control*: mask the same features on JetClass-II itself, re-run the probes, and
show the arm ordering is unchanged. That control is only worth anything if the
mask it applies is EXACTLY the fill the downstream config performs. Hand-writing
the two configs makes them drift the first time either is edited, and the drift
is invisible -- both files still parse, both still train, and the control
silently starts measuring a different intervention than the one it controls for.

So both are generated from ONE declaration, `FILLS` below, and
tests/test_downstream_fill.py re-derives the masked-slot sets from the emitted
YAML and asserts they are equal. The fill cannot drift from its control without
a test failing.

HOW THE FILL IS APPLIED
-----------------------
Not by redefining the missing branch -- weaver's `new_variables` shadowing a real
branch is undefined behaviour, and on the control the branch DOES exist. Instead
one variable

    part_zero: ak.zeros_like(part_energy)

is defined once, and each filled slot in `pf_features` reads `part_zero` while
KEEPING ITS ORIGINAL STANDARDIZATION PARAMETERS. That matters: the slot for
`part_d0err` is `[part_d0err, 0, 1, 0, 1]` (subtract 0, multiply 1, clip to
[0,1]) and the PID slots are `[..., null]` (no transform), so a raw 0 maps to a
network-input 0 under every one of them. Keeping the parameters therefore
changes nothing numerically and keeps the diff against the arm config to the
variable name alone, which is what makes it auditable.

WHAT IS *NOT* FILLED, AND WHY IT IS NOT A CHOICE
------------------------------------------------
The four kinematic slots and deta/dphi are present in every dataset. The fill
never touches them. `part_deltaR` is derived, not filled.

THE FILL IS A CONSTANT ACROSS ARMS. All 30 matrix arms and the MPM arm see the
identical masked input, so the fill cannot reorder them by construction; what it
can do is compress the contrast, because an arm that leans harder on
displacement loses more. That bias runs TOWARD the null, which is the safe
direction for a paper claiming a difference. Stated here because the control
measures the size of that compression and not its existence.

Run:  python3 scripts/build_downstream_configs.py [--check-only]
"""
from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "configs" / "finetune"
ARM_L162 = ROOT / "configs" / "arms" / "L162.yaml"
BASE = ROOT / "configs" / "data" / "JetClassII_base.yaml"

ZERO_PREFIX = "part_zero_"


def zero_name(var: str) -> str:
    """One zero variable per replaced feature, e.g. part_zero_d0err.

    NOT one shared `part_zero`. weaver keys standardization by variable NAME and
    refuses a name that appears with two different transforms
    (weaver/utils/data/config.py:94, "Incompatible info for variable"). Our
    filled slots carry two: the PID slots are `null` and the d0err/dzerr slots
    are `0, 1, 0, 1`. A single shared name therefore fails to load -- caught by
    experiments/FT/loadcheck.py before any GPU time.

    Per-feature names are better than the alternative of collapsing every filled
    slot onto one transform: each slot keeps the arms' own standardization, so
    the fill stays numerically identical to the unfilled pipeline, and the
    variable name records which feature it stands in for.
    """
    return ZERO_PREFIX + var.removeprefix("part_")

# The 17 pf_features slots, in the arms' order, with the arms' standardization.
# Re-derived from configs/arms/L162.yaml by tests/test_downstream_fill.py, which
# fails if this list drifts from the arm config.
FEATURES = [
    ("part_pt_scale_log", "1.7, 0.7"),
    ("part_e_scale_log", "2.0, 0.7"),
    ("part_logptrel", "-4.7, 0.7"),
    ("part_logerel", "-4.7, 0.7"),
    ("part_deltaR", "0.2, 4.0"),
    ("part_charge", "null"),
    ("part_isChargedHadron", "null"),
    ("part_isNeutralHadron", "null"),
    ("part_isPhoton", "null"),
    ("part_isElectron", "null"),
    ("part_isMuon", "null"),
    ("part_d0", "null"),
    ("part_d0err", "0, 1, 0, 1"),
    ("part_dz", "null"),
    ("part_dzerr", "0, 1, 0, 1"),
    ("part_deta", "null"),
    ("part_dphi", "null"),
]

PID = ["part_charge", "part_isChargedHadron", "part_isNeutralHadron",
       "part_isPhoton", "part_isElectron", "part_isMuon"]
DISPLACEMENT = ["part_d0", "part_d0err", "part_dz", "part_dzerr"]

FILLS = {
    "TopReference": dict(
        missing=PID + DISPLACEMENT,                       # 10 of 17
        # `is_signal_new` in the released file: 1 = top, 0 = QCD. Signal LAST so
        # class index 1 is the signal, which is what R50/R30 are quoted against.
        classes={"label_QCD": "label == 0", "label_Top": "label == 1"},
        source="Top Quark Tagging Reference set, Zenodo 2603256",
        why=("the released arrays carry only the constituent four-vectors "
             "(E, px, py, pz) and the binary label -- no particle type, no "
             "charge and no impact parameters exist to load"),
    ),
    "EnergyFlowQG": dict(
        missing=list(DISPLACEMENT),                       # 4 of 17
        # EnergyFlow's `y`: 1 = quark, 0 = gluon. Quark is the signal, so it is
        # index 1 for the same reason.
        classes={"label_gluon": "label == 0", "label_quark": "label == 1"},
        source="EnergyFlow quark/gluon, Zenodo 3164691 (Pythia)",
        why=("constituents carry (pt, y, phi, pdgid), so type and charge are "
             "recoverable in the 'exp' scenario, but the sample is generator "
             "level with no tracking and therefore no impact parameters"),
    ),
}

HEADER = """# {name}
# GENERATED by scripts/build_downstream_configs.py -- do not hand-edit.
#
# {source}
#
# ZERO-FILL: {n_missing} of the 17 per-particle features are absent from this
# dataset, because {why}. Each absent slot reads its own
# `part_zero_<feature>` (= ak.zeros_like(part_energy)) while keeping the arms'
# standardization parameters, under all of which a raw 0 maps to a network input
# of 0. One variable per feature and not one shared `part_zero`: weaver keys
# standardization by variable name and rejects a name used with two different
# transforms, and these slots carry two.
#
# Filled: {missing}
#
# The paired control is {control}, which masks the SAME
# slots on JetClass-II itself so the penalty is measurable in-domain. Both files
# come from one FILLS entry; tests/test_downstream_fill.py asserts the masked
# slot sets are equal.
#
# No `weights:` block: fine-tuning legs use the downstream sample at its natural
# composition (see scripts/build_finetune_configs.py for why).
"""

CONTROL_HEADER = """# {name}
# GENERATED by scripts/build_downstream_configs.py -- do not hand-edit.
#
# ZERO-FILL CONTROL for {partner}. This is the JetClass-II
# L162 fine-tuning config with EXACTLY the {n_missing} feature slots that
# {partner} must fill masked to zero, and nothing else
# changed. Running the frozen probes and one fine-tune per arm through this
# config measures what the fill costs, IN DOMAIN, where the unmasked number is
# also available -- which is the only place that subtraction can be made.
#
# Masked: {missing}
#
# What it can and cannot show: it bounds the penalty and shows whether the arm
# ORDERING survives masking. It does NOT prove the ordering survives on
# {partner}'s own distribution; nothing short of
# reduced-feature pretraining per target (~84 GPU-days, ParT's route) would, and
# that is the stated limitation rather than a gap the control quietly covers.
"""

BASE_CONTROL_HEADER = """# {name}
# GENERATED by scripts/build_downstream_configs.py -- do not hand-edit.
#
# ZERO-FILL CONTROL for {partner}, on the EXTRACTION side.
#
# WHY THIS EXISTS SEPARATELY FROM configs/finetune/JetClassII_L162_mask{short}.yaml.
# docs/PRD_PLAN.md 4.5 requires the control to re-run "the frozen probes AND one
# fine-tune per arm". Those two need DIFFERENT configs and the difference is not
# cosmetic:
#   - the fine-tune runs through the arm's own label vocabulary (L162 groups);
#   - the frozen probes run on features from experiments/EVAL/extract_features.py,
#     which every extraction spec runs with configs/data/JetClassII_base.yaml,
#     whose labels: block is `type: custom` -> truth_label: jet_label, the NATIVE
#     188-way label.
# The probes key on native labels (label_X_bb = 0, label_X_cc = 1,
# label_QCD_bb = 169). Extracting through the L162 control instead would hand
# them 162-GROUP labels under the same array name: every probe would still run,
# still return an AUC, and silently measure a different task.
#
# So this file is configs/data/JetClassII_base.yaml -- label block untouched --
# with EXACTLY the {n_missing} slots that {partner} fills
# masked to zero, and the weights: block dropped because an extraction pass does
# not reweight.
#
# Masked: {missing}
"""

NEW_VARIABLES_COMMON = """
new_variables:
{zeros}
   part_mask: ak.ones_like(part_energy)

   ## scaled vectors (jet rescaled so pT = 500 GeV) -- Sophon yaml:12-20
   part_px_scale: part_px / jet_pt * 500
   part_py_scale: part_py / jet_pt * 500
   part_pz_scale: part_pz / jet_pt * 500
   part_energy_scale: part_energy / jet_pt * 500

   part_pt: np.hypot(part_px, part_py)
   part_pt_scale: np.hypot(part_px_scale, part_py_scale)
   part_pt_scale_log: np.log(part_pt_scale)
   part_e_scale_log: np.log(part_energy_scale)
   part_logptrel: np.log(part_pt/jet_pt)
   part_logerel: np.log(part_energy/jet_energy)
   part_deltaR: np.hypot(part_deta, part_dphi)
"""


def _zero_defs(missing) -> str:
    return "".join(f"   {zero_name(v)}: ak.zeros_like(part_energy)\n" for v in missing)


def _feature_lines(missing) -> str:
    out = []
    for var, std in FEATURES:
        name = zero_name(var) if var in missing else var
        tail = f"   # zero-filled: {var}" if var in missing else ""
        out.append(f"         - [{name}, {std}]{tail}")
    return "\n".join(out)


def _inputs_block(missing) -> str:
    return f"""
inputs:
   pf_points:
      length: 128
      pad_mode: wrap
      vars:
         - [part_deta, null]
         - [part_dphi, null]
   pf_features:
      length: 128
      pad_mode: wrap
      vars:
{_feature_lines(missing)}
   pf_vectors:
      length: 128
      pad_mode: wrap
      vars:
         - [part_px_scale, null]
         - [part_py_scale, null]
         - [part_pz_scale, null]
         - [part_energy_scale, null]
   pf_mask:
      length: 128
      pad_mode: constant
      vars:
         - [part_mask, null]
"""


def downstream(name: str, spec: dict) -> str:
    missing = spec["missing"]
    head = HEADER.format(
        name=f"configs/finetune/{name}.yaml", source=spec["source"],
        n_missing=len(missing), why=spec["why"],
        missing=", ".join(missing),
        control=f"configs/finetune/JetClassII_L162_mask{name}.yaml")
    sel = ("selection:\n"
           "   ### Intentionally empty: the downstream sample is evaluated at its own\n"
           "   ### natural population. Porting Sophon's (200<pt<2500)&(20<msd<500) cut\n"
           "   ### would change what the benchmark row means.\n")
    cls = spec["classes"]
    # weaver's `type: simple` is a list of INDICATOR branches reduced with
    # np.argmax (weaver/utils/data/config.py:105-110), NOT a single integer
    # label. A one-element list would argmax over a width-1 axis and hand every
    # jet class 0 -- the loss would collapse, accuracy would read 1.0, and the
    # benchmark row would be meaningless without anything erroring. So the
    # integer label is expanded into one indicator per class here.
    ind = "".join(f"   {k}: {v}\n" for k, v in cls.items())
    return (head + "\n" + sel
            + NEW_VARIABLES_COMMON.format(zeros=_zero_defs(missing))
            + "\n   ## class indicators (weaver argmaxes over these)\n" + ind
            + "\npreprocess:\n  method: manual\n  data_fraction: 0.5\n"
            + _inputs_block(missing)
            + f"\nlabels:\n   type: simple\n   value: [{', '.join(cls)}]\n"
            + "\nobservers:\n   - jet_pt\n   - jet_energy\n")


def control(name: str, spec: dict, source_text: str, head: str, source_path=ARM_L162) -> str:
    """A source config, weights stripped, with exactly the fill's slots masked.

    Used twice per downstream set: once on the arm's L162 config (the fine-tune
    side) and once on configs/data/JetClassII_base.yaml (the extraction side,
    whose native 188-way label block the frozen probes require).
    """
    missing = spec["missing"]
    body = source_text
    # drop the weights: block -- the control is a fine-tuning/probe config
    i = body.find("\nweights:")
    if i != -1:
        j = body.find("\nlabels:", i)
        body = body[:i] + (body[j:] if j != -1 else "\n")
    # define the zero variable next to part_mask, which every config has
    body = body.replace("   part_mask: ak.ones_like(part_energy)",
                        _zero_defs(missing) + "\n   part_mask: ak.ones_like(part_energy)", 1)
    # mask exactly the declared slots, in place
    for var, std in FEATURES:
        if var not in missing:
            continue
        old = f"         - [{var}, {std}]"
        if old not in body:
            raise SystemExit(f"build_downstream_configs: slot {old!r} not found in "
                             f"{source_path} -- FEATURES has drifted from that config")
        body = body.replace(old, f"         - [{zero_name(var)}, {std}]   # masked: {var}", 1)
    return head + "\n" + body


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args()

    arm_text = ARM_L162.read_text()
    base_text = BASE.read_text()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stale = []
    for name, spec in FILLS.items():
        ft_head = CONTROL_HEADER.format(
            name=f"configs/finetune/JetClassII_L162_mask{name}.yaml",
            partner=f"configs/finetune/{name}.yaml",
            n_missing=len(spec["missing"]), missing=", ".join(spec["missing"]))
        base_head = BASE_CONTROL_HEADER.format(
            name=f"configs/finetune/JetClassII_base_mask{name}.yaml",
            partner=f"configs/finetune/{name}.yaml", short=name,
            n_missing=len(spec["missing"]), missing=", ".join(spec["missing"]))
        for path, text in (
                (OUT_DIR / f"{name}.yaml", downstream(name, spec)),
                (OUT_DIR / f"JetClassII_L162_mask{name}.yaml",
                 control(name, spec, arm_text, ft_head)),
                (OUT_DIR / f"JetClassII_base_mask{name}.yaml",
                 control(name, spec, base_text, base_head, source_path=BASE))):
            if a.check_only:
                if not path.exists() or path.read_text() != text:
                    stale.append(path.name)
            else:
                path.write_text(text)
                print(f"wrote {path.relative_to(ROOT)}  ({len(spec['missing'])} filled)")
    if a.check_only:
        if stale:
            print("STALE (re-run scripts/build_downstream_configs.py): "
                  + ", ".join(stale), file=sys.stderr)
            return 1
        print("configs/finetune downstream configs are up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
