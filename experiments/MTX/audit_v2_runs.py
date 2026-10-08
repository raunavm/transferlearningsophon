"""Audit every v2 grid run on /data against the rules that make the grid one experiment,
reading only what the runs wrote. Run in a pod (read-only):

    kubectl exec -i -n cms-ml <grid pod> -- python3 - < experiments/MTX/audit_v2_runs.py

Prints one line per check and a JSON summary (the last line). A check fails loudly; a
fact that is recorded but not a failure (for example which node trained an epoch) is
listed under "notes". Checks:
  epochs      metrics/ and stream/ hold epochs 0..N-1 with no gap, the stream record's
              sha256 is the one the epoch's metrics name, and n_jets is 10,240,000
  finite      every epoch's training and validation loss is finite
  qcd_share   every epoch's QCD share of the training stream (the v1 loader swung it
              8-18 %, amendment A7) within 0.5 % (absolute) of 14.436 %; a
              leave-one-family-out run (whose share is higher by design) within 0.5 % of
              its own median
  streams     within a run index, every run of the shared stream has the same stream
              sha256 at every epoch, and so has every leave-one-family-out run (A13 pairs them)
  lr          every epoch's learning rate is weaver's flat+decay: 5e-4 to epoch 55, then
              x 0.01^(1/24) per epoch (79 at 5e-6)
  fetch       every epoch's max_fetch_id is below 200, the fetches of one pass of its window
  init        within a run index, init_trunk.pt holds the same trunk for every run (A7),
              bit for bit, or else within 2 units in the last place of float32: the class
              token is drawn through erfinv, whose last bit depends on the node's CPU
              vector unit (found 2026-10-08: 1.86e-9 in 1-2 of 128 values, runs first
              started on ry-gpu-14), which is smaller than the rounding every training
              step on the GPU already adds; such runs are listed under notes
  recipe      recipe.json parses; within a run index runs differ only in the keys an arm
              may change; the seed is the run index; one code commit for the whole grid
  attempts    every attempt manifest names that commit, the run index's GPU product and
              one weights-block sha256 for the whole grid
  device      every epoch's metrics name the run index's GPU product
  resume      exactly one resume file, of the newest epoch, which loads and holds the
              model, optimizer, scheduler, scaler, trimmer counters and best epoch
  states      the state files retention keeps (EARLY_KEEP, the last ten, the best and the
              newest epoch) are there and load; no other state file, no .tmp debris
  best        best_epoch.json is the first maximum of the selection metric so far
  text        every .json in the run parses (no zero-filled file)
  limits      (notes) runs whose job would stop soon: NODE_FAULTS at 4 or more of the 6 that
              stop it, or a failed-attempt marker at the newest epoch (2 stop it)
"""
import glob
import hashlib
import json
import math
import os
import re
import sys

ROOT = "/data/results/mtx_v2"
REPO = "/workspace/transferlearningsophon"
EARLY_KEEP = (0, 2, 4, 9, 19, 29, 39, 49, 55, 62, 69)
NUM_EPOCHS, N_JETS, QCD_SHARE, QCD_TOL = 80, 10_240_000, 0.14436, 0.005
PRODUCT = {1: "NVIDIA GeForce RTX 3090", 2: "NVIDIA GeForce RTX 3090", 3: "NVIDIA GeForce RTX 3090",
           4: "NVIDIA L40", 5: "NVIDIA L40"}
# keys an arm may change between runs of one run index (everything else must agree)
ARM_KEYS = {"data_config", "network_config", "network_option", "extra_selection", "mass_lambda",
            "mpm", "mpm_mask_rate", "data_windows"}

fails, notes = [], []


def fail(check, msg):
    fails.append(f"{check}: {msg}")


def keep_epochs(best, newest):
    last10 = set(range(NUM_EPOCHS - 10, NUM_EPOCHS))
    return {e for e in EARLY_KEEP} | last10 | {best, newest}


def epoch_of(p, stem):
    return int(os.path.basename(p)[len("net_epoch-"):-len(stem)])


def main():
    import torch
    grid = {a["name"].lower().replace("_", ""): a for a in
            json.load(open(f"{REPO}/configs/arms/v2_grid.json"))["arms"]}
    runs = sorted(d for d in glob.glob(f"{ROOT}/mtx-*") if os.path.isdir(d))
    streams, inits, recipes, commits, wblocks = {}, {}, {}, set(), set()
    for d in runs:
        run = os.path.basename(d)
        arm, k = run[4:].rsplit("-s", 1)
        k = int(k)
        a = grid.get(arm)
        if a is None:
            fail("grid", f"{run} is not a grid run")
            continue
        lofo = bool(a.get("extra_selection"))
        # text: every json parses
        for f in glob.glob(f"{d}/**/*.json", recursive=True):
            try:
                json.load(open(f))
            except Exception as e:
                fail("text", f"{f}: {type(e).__name__}")
        # epochs, finite, qcd_share, device
        m = sorted(glob.glob(f"{d}/metrics/epoch-*.json"))
        eps = [int(f[-8:-5]) for f in m]
        if eps != list(range(len(eps))):
            fail("epochs", f"{run}: metrics epochs not contiguous from 0: {eps[:3]}..{eps[-3:]}")
        s_eps = sorted(int(f[-8:-5]) for f in glob.glob(f"{d}/stream/epoch-*.json"))
        if s_eps != eps:
            fail("epochs", f"{run}: stream epochs {len(s_eps)} != metrics epochs {len(eps)}")
        best_so_far, qs = None, []
        for f in m:
            e = int(f[-8:-5])
            r = json.load(open(f))
            sf = f"{d}/stream/epoch-{e:03d}.json"
            if os.path.exists(sf) and json.load(open(sf)).get("sha256") != r.get("stream_sha256"):
                fail("epochs", f"{run} epoch {e}: stream record sha256 != metrics stream_sha256")
            if r["train"].get("n_jets") != N_JETS:
                fail("epochs", f"{run} epoch {e}: n_jets {r['train'].get('n_jets')}")
            if not all(math.isfinite(v) for v in (r["train"]["loss"], r["val"]["loss"])):
                fail("finite", f"{run} epoch {e}: non-finite loss")
            q = r["train"].get("qcd_share")
            if q is None or (not lofo and abs(q - QCD_SHARE) > QCD_TOL):
                fail("qcd_share", f"{run} epoch {e}: {q}")
            qs.append(q)
            lr = 5e-4 * (0.01 ** (1 / 24)) ** max(0, e - 55)
            if abs(r["lr"] - lr) > 1e-9 * lr:
                fail("lr", f"{run} epoch {e}: {r['lr']} != {lr}")
            if not r["train"].get("max_fetch_id", 0) < 200:
                fail("fetch", f"{run} epoch {e}: max_fetch_id {r['train'].get('max_fetch_id')}")
            if r.get("device") != PRODUCT[k]:
                fail("device", f"{run} epoch {e}: {r.get('device')} (run index {k})")
            streams.setdefault((k, lofo, e), {})[run] = r["stream_sha256"]
            v = r["selection"]["value"]
            if best_so_far is None or v > best_so_far[1]:
                best_so_far = (e, v)
        if lofo and qs:
            med = sorted(qs)[len(qs) // 2]
            for e, q in enumerate(qs):
                if abs(q - med) > QCD_TOL:
                    fail("qcd_share", f"{run} epoch {e}: {q} (run median {med})")
        # limits
        nnf = sum(1 for _ in open(f"{d}/NODE_FAULTS")) if os.path.exists(f"{d}/NODE_FAULTS") else 0
        if nnf >= 4:
            notes.append(f"{run}: {nnf} NODE_FAULTS of the 6 that stop its job")
        if m and os.path.isdir(f"{d}/attempts"):
            nf = sum(1 for x in os.listdir(f"{d}/attempts") if x.endswith(f"-e{eps[-1]}"))
            if nf:
                notes.append(f"{run}: {nf} failed-attempt marker(s) at epoch {eps[-1]} (2 stop its job)")
        # best
        if m:
            b = json.load(open(f"{d}/best_epoch.json"))
            if b.get("epoch") != best_so_far[0]:
                fail("best", f"{run}: best_epoch.json {b.get('epoch')} != first maximum {best_so_far[0]}")
        # init
        ip = f"{d}/init_trunk.pt"
        if os.path.exists(ip):
            t = torch.load(ip, map_location="cpu", weights_only=False)["trunk"]
            h = hashlib.sha256()
            for name in sorted(t):
                h.update(name.encode())
                h.update(t[name].cpu().contiguous().numpy().tobytes())
            inits.setdefault(k, {})[run] = (h.hexdigest()[:16], ip)
        elif m:
            fail("init", f"{run}: no init_trunk.pt")
        # recipe (a run that never finished starting may have none; one with epochs must)
        if os.path.exists(f"{d}/recipe.json"):
            rec = json.load(open(f"{d}/recipe.json"))
            recipes.setdefault(k, {})[run] = rec
            if rec.get("seed") != k:
                fail("recipe", f"{run}: seed {rec.get('seed')} != run index {k}")
            commits.add(rec["code"]["commit"])
        elif m:
            fail("recipe", f"{run}: {len(m)} epochs and no recipe.json")
        else:
            notes.append(f"{run}: no recipe.json and no epochs (never started training): {sorted(os.listdir(d))}")
        # attempts
        for mf in glob.glob(f"{d}/run_manifest*.json"):
            man = json.load(open(mf))
            h = man.get("hardware", {})
            prod = (h.get("gpu_device_name") or "")
            if prod and prod != PRODUCT[k]:
                fail("attempts", f"{run} {os.path.basename(mf)}: GPU {prod}")
            prov = man.get("provenance", {})
            c = prov.get("repo_commit")
            if c:
                commits.add(c)
            else:
                fail("attempts", f"{run} {os.path.basename(mf)}: no provenance.repo_commit")
            wb = prov.get("weights_block_sha256")
            if wb:
                wblocks.add(wb)
            node = (h.get("node_name") or "").split(".")[0]
            if node in ("ry-gpu-04", "ry-gpu-09", "nrp-01"):
                notes.append(f"{run}: an attempt ran on {node} ({os.path.basename(mf)})")
        # resume and states
        if not m:
            continue
        newest = eps[-1]
        res = glob.glob(f"{d}/net_epoch-*_resume.pt")
        if [epoch_of(p, "_resume.pt") for p in res] != [newest]:
            fail("resume", f"{run}: resume files {sorted(os.path.basename(p) for p in res)}, newest epoch {newest}")
        for p in res:
            try:
                st = torch.load(p, map_location="cpu", weights_only=False)
                miss = {"epoch", "model", "optimizer", "scheduler", "scaler", "trimmer_counters", "best"} - set(st)
                if miss or st["epoch"] != epoch_of(p, "_resume.pt"):
                    fail("resume", f"{run}: {os.path.basename(p)} missing {sorted(miss)} or wrong epoch")
            except Exception as e:
                fail("resume", f"{run}: {os.path.basename(p)} does not load: {type(e).__name__}")
        want = {e for e in keep_epochs(best_so_far[0], newest) if e <= newest}
        have = {epoch_of(p, "_state.pt") for p in glob.glob(f"{d}/net_epoch-*_state.pt")
                if not p.endswith("_bn_state.pt")}
        for p in glob.glob(f"{d}/net_epoch-*_bn_state.pt"):      # BatchNorm twins (bn_twins_v2.py)
            try:
                torch.load(p, map_location="cpu", weights_only=False)
            except Exception as ex:
                fail("states", f"{run}: {os.path.basename(p)} does not load: {type(ex).__name__}")
        if want - have:
            fail("states", f"{run}: missing state files for epochs {sorted(want - have)}")
        if have - want:
            fail("states", f"{run}: unexpected state files for epochs {sorted(have - want)}")
        for e in sorted(have):
            try:
                torch.load(f"{d}/net_epoch-{e}_state.pt", map_location="cpu", weights_only=False)
            except Exception as ex:
                fail("states", f"{run}: net_epoch-{e}_state.pt does not load: {type(ex).__name__}")
        for t in glob.glob(f"{d}/**/*.tmp", recursive=True):
            fail("states", f"{run}: leftover {t}")
    # cross-run checks
    for (k, lofo, e), dd in sorted(streams.items()):
        if len(set(dd.values())) > 1:
            fail("streams", f"run index {k}{' (leave-one-family-out)' if lofo else ''} epoch {e}: {dd}")
    eps32 = float(torch.finfo(torch.float32).eps)
    for k, dd in sorted(inits.items()):
        hashes = [h for h, _ in dd.values()]
        major = max(set(hashes), key=hashes.count)
        ref = torch.load(next(p for h, p in dd.values() if h == major), map_location="cpu",
                         weights_only=False)["trunk"]
        for run, (h, ip) in sorted(dd.items()):
            if h == major:
                continue
            t = torch.load(ip, map_location="cpu", weights_only=False)["trunk"]
            worst = []
            for name in ref:
                a, b = ref[name].double(), t[name].double()
                if a.shape != b.shape:
                    worst.append((name, float("inf")))
                    continue
                ulp = (a - b).abs() / (eps32 * a.abs().clamp_min(1e-30))
                if ulp.max() > 0:
                    worst.append((name, float(ulp.max()), int((ulp > 0).sum()), float((a - b).abs().max())))
            if any(w[1] > 2 for w in worst):
                fail("init", f"run index {k}: {run} trunk differs from the run index's: {worst[:3]}")
            else:
                notes.append(f"{run}: init trunk within float32 rounding of run index {k}'s "
                             f"(tensors, max ulp, values, max |diff|): {worst}")
    for k, dd in sorted(recipes.items()):
        names = sorted(dd)
        for r in names[1:]:
            diff = {key for key in set(dd[names[0]]) | set(dd[r])
                    if dd[names[0]].get(key) != dd[r].get(key)} - ARM_KEYS
            if diff:
                fail("recipe", f"run index {k}: {names[0]} and {r} differ in {sorted(diff)}")
    if len(commits) != 1:
        fail("recipe", f"code commits {sorted(commits)}")
    if len(wblocks) > 1:
        fail("attempts", f"weights-block sha256 {sorted(wblocks)}")
    n_shared = sum(1 for dd in streams.values() if len(dd) > 1)
    for f in fails:
        print("FAIL", f)
    for n in notes:
        print("NOTE", n)
    summary = {"runs": len(runs), "epochs": sum(1 for _ in glob.glob(f"{ROOT}/mtx-*/metrics/epoch-*.json")),
               "shared_stream_slots": n_shared, "init_bit_identical_groups": {k: len({h for h, _ in v.values()}) for k, v in inits.items()},
               "commits": sorted(commits), "weights_blocks": len(wblocks), "fails": len(fails), "notes": len(notes)}
    print(json.dumps(summary))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
