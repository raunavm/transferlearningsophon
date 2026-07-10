# E1 — Pipeline validation & preprocessing tax (RUNBOOK)

Train the **Sophon architecture** from scratch on **JetClass-1** twice, differing only in preprocessing:
**Arm P** (ParT preprocessing) and **Arm S** (Sophon preprocessing). Gate: Arm P macro AUC within ~0.002 of published ParT (0.9877). Tax = Arm P − Arm S.

E1 runs through **weaver directly** (the official framework that produced the published numbers), NOT the repo's `train_finetune_sweep.py`. It reads raw ROOT from `/data/JetClass/Pythia/{train_100M,val_5M,test_20M}` (already on the PVC).

Read `EXTRACTED_GROUND_TRUTH.md` first — every hyperparameter is verified there with `file:line` citations and a drift table.

## Artifacts in this directory
- `data_arm_p.yaml` — Arm P config (verbatim ParT `JetClass_full.yaml`).
- `data_arm_s.yaml` — Arm S config (Sophon preprocessing ported to JetClass-1; **reweight bins provisional until the probe job confirms them**).
- `ParT_sophon_arch_10c.py` — network config, Sophon arch, 10 classes, custom train/eval removed (landmine fix). Used by BOTH arms.
- `probe_jetclass1.py` — data-integrity + (jet_pt, jet_sdmass) inspection to finalize Arm S bins.
- `eval_e1.py` — metrics (macro/per-class AUC, rejection@eff, TPR@FPR, bootstrap CIs, paired tax) from weaver's `pred.root`.
- `k8s/` — job manifests, one per phase.
- `E1_REPORT.md` — fill in as results land.

## Order of operations (each cluster launch needs explicit go-ahead)

**Step 0 — publish the configs (user runs; I never commit/push).**
The cluster jobs clone `github.com/raunavm/transferlearningsophon`, so `experiments/E1/` must be pushed first:
```
git add experiments/E1 && git commit -m "E1: configs + manifests" && git push
```

**Step 1 — PROBE (read-only; ~minutes).** `k8s/job-e1-probe-raunav.yaml`
- Records weaver version + `weaver -h` (confirms CLI + `flat+decay` default; decides if a pinned reinstall is needed — see D6).
- `weaver --print` on BOTH YAMLs over 1 file/class: confirms configs parse, the network file loads at `num_classes=10`, prints param/FLOP count, exercises label handling — **no training**.
- Runs `probe_jetclass1.py`: per-class entry counts (integrity), (jet_pt, jet_sdmass) min/max/quantiles, and fraction of jets outside the provisional Arm S bins.
- Output → `/data/results/e1/probe/`.
- **Gate:** all 10 classes present with expected counts; <0.5% of jets outside bins. If not, edit `data_arm_s.yaml` bin edges from the probe report and re-run the probe.

**Step 2 — SMOKE / dry run (~30 min, 1 GPU).** `k8s/job-e1-smoke-raunav.yaml`
- Both arms, 2 epochs, 1 file/class train, tiny val — confirms: loss decreases, no NaN under AMP, Arm S `make_weight` produces a sane weight distribution (max/min ratio logged), checkpoint+resume works (kill/resume once), throughput → **projected full-run wall time** (report before launching full training; get sign-off if >7 GPU-days/arm).
- Produces a tiny `pred.root` — **inspect its branch names to lock `eval_e1.py`'s schema** before the full eval.

**Step 3 — FULL TRAIN, both arms (~days, 1 GPU each).** `k8s/job-e1-train-raunav.yaml`
- Single-GPU official recipe (no DDP): `--batch-size 512 --start-lr 1e-3 --samples-per-epoch 10240000 --samples-per-epoch-val 1280000 --num-epochs 50 --optimizer ranger --num-workers 2 --fetch-step 0.01 --use-amp`, `flat+decay` (default), best = highest val accuracy.
- Launch Arm P and Arm S as two jobs (edit `ARM` + name). Stagger 60–90 s. Repeat each arm a 2nd time for seed spread (no `--seed` in weaver — D1).
- Monitor: val-accuracy/epoch, LR-schedule shape (verify 70%-flat then decay from the log), GPU util. Do NOT touch test.

**Step 4 — EVAL (after both best checkpoints frozen; ~1 h, 1 GPU).** `k8s/job-e1-eval-raunav.yaml`
- `weaver --predict` with `--model-prefix .../net_best_epoch_state.pt` on `test_20M` → `pred.root` per arm.
- `eval_e1.py` → macro AUC, per-class AUC, ParT-convention rejection (discriminant `p_S/(p_S+p_QCD)`) at ε=0.5/0.3, TPR@FPR 1e-3/1e-4, bootstrap CIs (≥200), paired tax Arm P−Arm S with CIs.

**Step 5 — GATE + report.** Fill `E1_REPORT.md`. If Arm P misses 0.002, diagnose in the order in `EXTRACTED_GROUND_TRUTH.md` §8 (start with the plain-ParT-network control). Never fix a gap by changing evaluation.

## Multi-GPU (optional speedup)
Wrap weaver in `torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS $(which weaver) --backend nccl`, set `--samples-per-epoch $((10240000/NGPUS))`, and **scale `--start-lr` linearly** (2 GPU → 2e-3). Keep NGPUS identical across arms. Single-GPU is the default for foolproofness.
