"""The MLST paper's inputs from /data, as small JSONs for experiments/FIGS/data/v2/.

Runs in one of our pods (it reads /data) and writes a gzipped tar to stdout:

  linprobe/fits3000/<model>.json   the linear-probe fits, unchanged (scripts/build_linprobe_jobs.py)
  linprobe/mass/<model>.json       the jet-mass regressions, unchanged (mass_resolution.json)
  compute/<run>.json               per epoch of every finished run: the GPU it trained on and
                                   the seconds of its training and validation passes, read from
                                   <run>/metrics/epoch-NNN.json

Only finished runs (a DONE marker) are taken, so a run is never read half-trained.

    kubectl exec -i -n cms-ml <pod> -- python3 - < experiments/FIGS/pull_mlst_data.py \\
        | tar xzf - -C experiments/FIGS/data/v2
"""
import io
import json
import pathlib
import sys
import tarfile

LP = pathlib.Path("/data/results/eval/v2_linprobe")
MTX = pathlib.Path("/data/results/mtx_v2")
N_EPOCHS = 80


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
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        def add(name: str, data: bytes):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        for f in sorted((LP / "fits3000").glob("*.json")):
            add(f"linprobe/fits3000/{f.name}", f.read_bytes())
        for f in sorted((LP / "mass").glob("*/mass_resolution.json")):
            add(f"linprobe/mass/{f.parent.name}.json", f.read_bytes())
        for run in sorted(MTX.glob("mtx-*")):
            if (run / "DONE").exists():
                add(f"compute/{run.name}.json", (json.dumps(compute(run), indent=1) + "\n").encode())
    sys.stdout.buffer.write(buf.getvalue())
    return 0


if __name__ == "__main__":
    sys.exit(main())
