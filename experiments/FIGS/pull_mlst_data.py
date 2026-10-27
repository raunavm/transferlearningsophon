"""The MLST paper's inputs from /data, as small JSONs for experiments/FIGS/data/v2/.

Runs in one of our pods (it reads /data) and writes a gzipped tar to stdout:

  linprobe/fits3000/<model>.json   the linear-probe fits, unchanged (scripts/build_linprobe_jobs.py)
  linprobe/mass/<model>.json       the jet-mass regressions, unchanged (mass_resolution.json)
  compute/<run>.json               per epoch of every finished run: the GPU it trained on and
                                   the seconds of its training and validation passes, read from
                                   <run>/metrics/epoch-NNN.json
  ft_mlst/<tag>_<file>.json        the fine-tuning read-outs (scripts/build_ft_jobs.py, v2 read-outs):
                                   <tag> best70 (every paper run) or best70_done18 (the 18 runs
                                   finished before the split), <file> leg1_metrics, leg2_metrics,
                                   bench_metrics, bench_metrics_herwig; each must parse
  ft_mlst/manifests.json           {cell: its recipe, GPU and node} for every DONE fine-tuning cell of
                                   the paper runs (best70) and the V100 references (ref_v100), read
                                   from each cell's ft_manifest.json
  index.json                       {path: sha256} of every file above: the single source the
                                   macros counted over many files cite (mlst_results.py checks it)

Only finished runs (a DONE marker) are taken, so a run is never read half-trained.

    kubectl exec -i -n cms-ml <pod> -- python3 - < experiments/FIGS/pull_mlst_data.py \\
        | tar xzf - -C experiments/FIGS/data/v2
"""
import hashlib
import io
import json
import pathlib
import sys
import tarfile

LP = pathlib.Path("/data/results/eval/v2_linprobe")
FT = pathlib.Path("/data/results/ft_v2")
FT_TAGS = ("best70", "best70_done18")
FT_FILES = {"leg1": ("leg1_metrics",), "leg2": ("leg2_metrics",),
            "bench": ("bench_metrics", "bench_metrics_herwig")}
MTX = pathlib.Path("/data/results/mtx_v2")
N_EPOCHS = 80
MANIFEST_KEYS = ("leg", "init", "n_train", "ft_seed", "lr", "head_lr_mult", "weight_decay", "epochs",
                 "lr_schedule", "recipe", "batch_size", "samples_per_epoch", "ckpt_rule",
                 "gpu_device_name", "node_name", "repo_commit", "checkpoint_sha256")


def manifests() -> dict:
    out = {}
    for pattern in ("best70/*/*/N*/s*", "ref_v100/*/*/*/N*/s*"):
        for cell in sorted(FT.glob(pattern)):
            if cell.is_dir() and ".partial." not in cell.name and not cell.name.endswith(".lock") \
                    and (cell / "DONE").exists():
                m = json.loads((cell / "ft_manifest.json").read_text())
                out[str(cell.relative_to(FT))] = {k: m.get(k) for k in MANIFEST_KEYS}
    return out


def compute(run: pathlib.Path) -> dict:
    epochs = []
    for e in range(N_EPOCHS):
        d = json.loads((run / "metrics" / f"epoch-{e:03d}.json").read_text())
        if d["epoch"] != e or "device" not in d:
            raise SystemExit(f"FATAL: {run.name} epoch {e}: no device record")
        epochs.append({"epoch": e, "device": d["device"], "train_seconds": d["train"]["seconds"],
                       "val_seconds": d["val"]["seconds"], "jets_per_s": d["train"]["jets_per_s"]})
    return {"run": run.name, "epochs": epochs}


def main() -> int:
    buf = io.BytesIO()
    index = {}
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        def add(name: str, data: bytes):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
            index[name] = hashlib.sha256(data).hexdigest()
        for f in sorted((LP / "fits3000").glob("*.json")):
            add(f"linprobe/fits3000/{f.name}", f.read_bytes())
        for f in sorted((LP / "mass").glob("*/mass_resolution.json")):
            add(f"linprobe/mass/{f.parent.name}.json", f.read_bytes())
        for tag in FT_TAGS:
            for leg, names in FT_FILES.items():
                for n in names:
                    f = FT / f"{tag}_{leg}_metrics" / f"{n}.json"
                    if f.exists():
                        b = f.read_bytes()
                        json.loads(b)                       # a file still being written stops the pull
                        add(f"ft_mlst/{tag}_{n}.json", b)
        add("ft_mlst/manifests.json", (json.dumps(manifests(), indent=1) + "\n").encode())
        for run in sorted(MTX.glob("mtx-*")):
            if (run / "DONE").exists():
                add(f"compute/{run.name}.json", (json.dumps(compute(run), indent=1) + "\n").encode())
        add("index.json", (json.dumps(dict(sorted(index.items())), indent=1) + "\n").encode())
    sys.stdout.buffer.write(buf.getvalue())
    return 0


if __name__ == "__main__":
    sys.exit(main())
