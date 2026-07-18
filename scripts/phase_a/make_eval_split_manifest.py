#!/usr/bin/env python3
"""Phase A A.3: freeze the JetClass-II evaluation split (G0 artifact, PI-visible).

Designation (per PLAN §4, audit recommendation): the release **test** partition
(335 files) is the confirmatory evaluation population; the release **val**
partition (335 files) is reserved for ALL selection (probe C-grids, early
stopping, checkpoint choice). P2/P3 blinding semantics apply to `test`.

The release has no per-event UID columns (verified 2026-07-17: the public
ntupler defines none), so the frozen identity is the file list itself:
family + 4-digit index ranges under the canonical 4:1:1 file-index split.
The in-cluster stats-preflight job verifies every listed file exists on the
PVC and records jet counts + the sha256 of the sorted realized file list.

Run from the repo root:  python3 scripts/phase_a/make_eval_split_manifest.py
Writes: manifests/release_eval_split.json
"""
from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# family -> (total files, train/val/test file-index ranges) — canonical 4:1:1
FAMILIES = {
    "Res2P":  {"total": 300,  "train": (0, 199),   "val": (200, 249),   "test": (250, 299)},
    "Res34P": {"total": 1290, "train": (0, 859),   "val": (860, 1074),  "test": (1075, 1289)},
    "QCD":    {"total": 420,  "train": (0, 279),   "val": (280, 349),   "test": (350, 419)},
}


def files_of(part):
    out = []
    for fam, spec in FAMILIES.items():
        lo, hi = spec[part]
        out += [f"{fam}_{i:04d}.parquet" for i in range(lo, hi + 1)]
    return sorted(out)


def main():
    parts = {p: files_of(p) for p in ("train", "val", "test")}
    counts = {p: len(v) for p, v in parts.items()}
    assert counts == {"train": 1340, "val": 335, "test": 335}, counts
    list_hash = {p: hashlib.sha256("\n".join(v).encode()).hexdigest()
                 for p, v in parts.items()}

    out = {
        "artifact": "PLAN v5.1 §4 / §6-A.3 frozen evaluation split  ⟨BIND:G0⟩",
        "frozen_on": str(date.today()),
        "designation": {
            "confirmatory_eval": "test",
            "selection_only": "val",
            "rationale": ("v5.1 audit recommendation: release-test as the "
                          "confirmatory population, release-val reserved for all "
                          "selection (cleaner P2 hygiene). PI sign-off: PENDING "
                          "— recorded as the working designation until countersigned."),
        },
        "split_rule": "canonical 4:1:1 by file index per family",
        "family_index_ranges": FAMILIES,
        "file_counts": counts,
        "file_list_sha256": list_hash,
        "test_files": parts["test"],
        "val_files": parts["val"],
        "event_uid_note": ("Release parquet carries no per-event UID columns "
                           "(ntupler defines none; verified 2026-07-17). Frozen "
                           "identity = this file list; per-jet row order within "
                           "files is the release order. The G corpus (§7) will "
                           "carry true event UIDs via its L5 stamp."),
        "verification": ("scripts/phase_a/stats_preflight.py checks existence of "
                         "every listed file on the PVC and records realized jet "
                         "counts + sorted-file-list sha256 in its report."),
    }
    dest = REPO / "manifests/release_eval_split.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {dest}")
    print("test files:", counts["test"], "| val files:", counts["val"],
          "| test-list hash:", list_hash["test"][:16], "…")


if __name__ == "__main__":
    main()
