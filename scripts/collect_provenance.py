#!/usr/bin/env python3
"""Collect a per-run provenance record for the paper.

Reconstructs everything reportable from a weaver run's train.log plus a few
passed-in facts (GPU model/count, node, seed, avg power), computes derived
compute figures (GPU-hours, FLOPs, energy), writes <run-dir>/run_provenance.json,
and prints a markdown block ready to paste/append into experiments/RESULTS.md.

Works retroactively (parses a finished or in-flight train.log) and live.
FLOPs need the network config + torch; if fvcore is unavailable it is left null.
"""
import argparse
import datetime as dt
import json
import re
import subprocess
import sys
from pathlib import Path

# approximate board TDP (W) for an energy estimate when avg power isn't measured
TDP_W = {
    "NVIDIA-L40": 300, "NVIDIA-L40S": 350, "NVIDIA-A100-SXM4-80GB": 400,
    "NVIDIA-A100-PCIE-40GB": 250, "NVIDIA-A100-80GB-PCIe": 300, "NVIDIA-A40": 300,
    "NVIDIA-RTX-A6000": 300, "NVIDIA-GeForce-RTX-4090": 450, "NVIDIA-A10": 150,
    "NVIDIA-GeForce-RTX-3090": 350, "NVIDIA-TITAN-RTX": 280,
}
GRID_KG_PER_KWH = 0.371  # US avg (EPA eGRID); note in paper if used


def _ts(line):
    m = re.match(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+\]", line)
    return dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S") if m else None


def parse_log(path):
    txt = Path(path).read_text(errors="ignore").splitlines()
    args, in_args = {}, False
    params_m, epochs, first_ts, last_ts = None, [], None, None
    for ln in txt:
        t = _ts(ln)
        if t:
            first_ts = first_ts or t
            last_ts = t
        if "INFO: args:" in ln:
            in_args = True
            continue
        if in_args:
            m = re.match(r"\s*-\s*\('([^']+)',\s*(.*)\)\s*$", ln)
            if m:
                args[m.group(1)] = m.group(2).strip()
            elif ln.strip() and not ln.strip().startswith("- ("):
                in_args = False
        pm = re.search(r"Number of parameters:\s*([\d.]+)\s*M", ln)
        if pm:
            params_m = float(pm.group(1))
        em = re.search(r"Epoch #(\d+): Current validation metric: ([\d.]+) \(best: ([\d.]+)\)", ln)
        if em:
            epochs.append({"epoch": int(em.group(1)), "val": float(em.group(2)),
                           "best": float(em.group(3)), "ts": _ts(ln)})
    return args, params_m, epochs, first_ts, last_ts


def versions():
    import platform
    v = {"python": platform.python_version()}
    try:
        import torch
        v.update(torch=torch.__version__, cuda=torch.version.cuda,
                 cudnn=torch.backends.cudnn.version())
    except Exception:
        pass
    for m in ("weaver", "numpy", "uproot", "awkward", "sklearn"):
        try:
            v[m] = __import__(m).__version__
        except Exception:
            pass
    return v


def checkpoints(run_dir):
    ck = sorted(Path(run_dir).glob("net_epoch-*_state.pt"))
    size_gb = round(sum(p.stat().st_size for p in Path(run_dir).glob("net*.pt")) / 1e9, 2)
    return {"n_epoch_checkpoints": len(ck), "total_ckpt_size_gb": size_gb,
            "best_ckpt": str(Path(run_dir) / "net_best_epoch_state.pt")}


def driver_version():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version",
                              "--format=csv,noheader"], capture_output=True, text=True)
        return out.stdout.strip().splitlines()[0] if out.stdout.strip() else None
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", default="UNSEEDED (weaver 0.4.17 has no --seed)")
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--gpu-count", type=int, default=1)
    ap.add_argument("--node", default="")
    ap.add_argument("--driver", default="")
    ap.add_argument("--avg-power-w", type=float, default=None)
    ap.add_argument("--git-commit", default="")
    ap.add_argument("--image-digest", default="")
    ap.add_argument("--net-config", default="")
    ap.add_argument("--num-classes", type=int, default=10)
    ap.add_argument("--flops-json", default="", help="path to flops.json (macs_per_jet_fwd)")
    ap.add_argument("--results-md", default="")
    args = ap.parse_args()

    log = Path(args.run_dir) / "train.log"
    hp, params_m, epochs, t0, t1 = parse_log(log)

    n_done = len(epochs)
    spe = int(hp.get("samples_per_epoch", "0").replace("None", "0") or 0)
    wall_s = (t1 - t0).total_seconds() if (t0 and t1) else None
    per_ep_min = round(wall_s / max(n_done, 1) / 60, 1) if wall_s else None
    gpu_h = round((wall_s / 3600) * args.gpu_count, 2) if wall_s else None
    samples_seen = n_done * spe
    throughput = round(samples_seen / wall_s, 1) if wall_s else None
    macs = None
    if args.flops_json and Path(args.flops_json).exists():
        macs = json.loads(Path(args.flops_json).read_text()).get("macs_per_jet_fwd")
    total_train_flops = (macs * 2 * samples_seen * 3) if macs else None  # 2xMAC, fwd+bwd≈3x
    pwr_kw = (args.avg_power_w or TDP_W.get(args.gpu, 300)) / 1000.0
    energy_kwh = round(gpu_h * pwr_kw, 2) if gpu_h else None
    # weaver's running "best" is monotonic in the monitor direction, so the
    # final epoch's value is the overall extremum whether higher- or lower-is-
    # better; the best epoch is the first one that reached it (direction-agnostic).
    _final_best = epochs[-1]["best"] if epochs else None
    best = next((e for e in epochs if e["best"] == _final_best), {}) if epochs else {}

    rec = {
        "arm": args.arm, "seed": args.seed, "run_dir": args.run_dir,
        "git_commit": args.git_commit, "image_digest": args.image_digest,
        "versions": versions(),
        "hardware": {"gpu": args.gpu, "gpu_count": args.gpu_count, "node": args.node,
                     "driver": args.driver or driver_version()},
        "compute": {
            "params": int(params_m * 1e6) if params_m else None, "params_M": params_m,
            "macs_per_jet_fwd": macs, "flops_per_jet_2xmac": (2 * macs) if macs else None,
            "part_reference_macs": 340_000_000,
            "epochs_completed": n_done, "samples_per_epoch": spe, "samples_seen": samples_seen,
            "throughput_jets_per_s": throughput,
            "wall_clock_h": round(wall_s / 3600, 2) if wall_s else None,
            "per_epoch_min": per_ep_min, "gpu_hours": gpu_h,
            "total_training_flops_approx": total_train_flops,
            "flops_convention": "MACs=fvcore (ParT-comparable); total=2xMAC x samples x 3 (fwd+bwd)",
            "energy_kwh_est": energy_kwh,
            "co2e_kg_est": round(energy_kwh * GRID_KG_PER_KWH, 2) if energy_kwh else None,
            "energy_method": "measured avg power" if args.avg_power_w else f"TDP estimate ({TDP_W.get(args.gpu,300)}W)",
            "grid_factor_kg_per_kwh": GRID_KG_PER_KWH,
        },
        "hyperparameters": {k: hp.get(k) for k in (
            "optimizer", "optimizer_option", "lr_scheduler", "start_lr", "batch_size",
            "num_epochs", "samples_per_epoch", "samples_per_epoch_val", "num_workers",
            "fetch_step", "use_amp", "load_epoch", "network_option", "data_config", "network_config")},
        "full_weaver_args": hp,   # everything weaver was invoked with (extra, for our reference)
        "checkpoints": checkpoints(args.run_dir),
        "selection": {"best_epoch": best.get("epoch"), "best_val_metric": best.get("best")},
        "val_curve": [{"epoch": e["epoch"], "val": e["val"]} for e in epochs],
    }
    out = Path(args.run_dir) / "run_provenance.json"
    out.write_text(json.dumps(rec, indent=2))
    print(f"[provenance] wrote {out}")

    c, hpx = rec["compute"], rec["hyperparameters"]
    tf = c["total_training_flops_approx"]
    tf_str = f"{tf:.2e} (~{tf/1e18:.1f} EFLOP)" if tf else "n/a"
    md = (f"\n### {args.arm}  (seed={args.seed})\n"
          f"- **Result**: best val metric {rec['selection']['best_val_metric']} "
          f"@ epoch {rec['selection']['best_epoch']}\n"
          f"- **Hardware**: {args.gpu} ×{args.gpu_count}"
          + (f" ({args.node})" if args.node else "") + "\n"
          f"- **Compute**: {c['params_M']} M params · {c['epochs_completed']} epochs · "
          f"{c['samples_seen']:,} jets seen · {c['wall_clock_h']} h wall "
          f"({c['per_epoch_min']} min/ep) · {c['gpu_hours']} GPU-h · "
          f"{c['throughput_jets_per_s']} jets/s\n"
          f"- **FLOPs**: {round((c['macs_per_jet_fwd'] or 0)/1e6,1)} M MACs/jet fwd "
          f"(ParT ref 340 M) · {tf_str} total training FLOPs (approx)\n"
          f"- **Energy**: ~{c['energy_kwh_est']} kWh / ~{c['co2e_kg_est']} kg CO2e "
          f"[{c['energy_method']}]\n"
          f"- **Config**: {hpx['optimizer']} + {hpx['lr_scheduler']}, lr {hpx['start_lr']}, "
          f"batch {hpx['batch_size']}, AMP {hpx['use_amp']}, data `{hpx['data_config']}`\n"
          f"- **Versions**: weaver {rec['versions'].get('weaver')}, "
          f"torch {rec['versions'].get('torch')}, cuda {rec['versions'].get('cuda')}\n"
          f"- **git** `{args.git_commit}` · **image** `{args.image_digest[-16:]}`\n")
    print("=== MARKDOWN BLOCK ===")
    print(md)
    if args.results_md:
        with open(args.results_md, "a") as f:
            f.write(md)
        print(f"[provenance] appended to {args.results_md}")


if __name__ == "__main__":
    main()
