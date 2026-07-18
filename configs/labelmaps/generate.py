#!/usr/bin/env python3
"""Emit configs/labelmaps/*.yaml verbatim from the frozen label_maps.json.

PLAN §2 legacy map + §9.1: the portable per-arm label maps are generated from
`experiments/E2/labels/label_maps.json` (the frozen source of truth), with the
source SHA256 and each emitted file's SHA256 recorded in manifest.json. Rerunning
is byte-identical. No map is ever regenerated from a seed here.
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent / "experiments" / "E2" / "labels" / "label_maps.json"

# arm key in JSON -> emitted filename stem (PLAN §2 configs/labelmaps/ names)
ARMS = {
    "g2": "k2",
    "g10sem": "k10_semantic",
    "g10rand": "k10_random_draw1",
    "g30": "k30",
    "g188": "k188",
}


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def yaml_scalar(v):
    return f'"{v}"' if isinstance(v, str) else str(v)


def main():
    src_bytes = SRC.read_bytes()
    data = json.loads(src_bytes)
    names188 = data["class_names_188"]
    manifest = {
        "source": "experiments/E2/labels/label_maps.json",
        "source_sha256": sha256(src_bytes),
        "generator": "configs/labelmaps/generate.py",
        "files": {},
    }
    for arm_key, stem in ARMS.items():
        m = data[arm_key]
        lut = m["map"]
        assert len(lut) == 188
        lines = [
            f"# {stem} — {m['num_classes']}-class label map for JetClass-II (188 -> {m['num_classes']}).",
            f"# {m['description']}",
            f"# Generated verbatim from label_maps.json (sha256 {manifest['source_sha256'][:16]}…);",
            "# the JSON is the source of truth. Do not hand-edit.",
            f"name: {stem}",
            f"num_classes: {m['num_classes']}",
            "class_names:",
        ]
        lines += [f"  - {yaml_scalar(c)}" for c in m["class_names"]]
        lines.append("# map[i] = merged class index for native jet_label i (i = 0..187)")
        lines.append("map:")
        lines += [f"  - {v}   # {names188[i]}" for i, v in enumerate(lut)]
        text = "\n".join(lines) + "\n"
        out = HERE / f"{stem}.yaml"
        out.write_text(text)
        manifest["files"][f"{stem}.yaml"] = {
            "num_classes": m["num_classes"],
            "sha256": sha256(text.encode()),
        }
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for f, meta in manifest["files"].items():
        print(f"{f:24s} {meta['num_classes']:3d} classes  sha256 {meta['sha256'][:16]}…")
    print(f"source label_maps.json sha256 {manifest['source_sha256']}")


if __name__ == "__main__":
    main()
