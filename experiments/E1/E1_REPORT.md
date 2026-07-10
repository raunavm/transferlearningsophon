# E1 Report — Pipeline validation & preprocessing tax

*Fill in as results land. External reference rows (published ParT/Sophon) are lines, never comparison bars.*

## Setup
- Pinned commits: ParT `2925bdb`, Sophon `9dd6dd6`, weaver `c97de3c` (dev/custom_train_eval). See `EXTRACTED_GROUND_TRUTH.md`.
- Environment: image `escheuller/transfer-learning:cu121`; weaver version recorded by probe → `/data/results/e1/probe/weaver_version.txt`. `pip freeze` → attach as `environment.lock`.
- Architecture (both arms): Sophon ParT, 8 particle-attention + 2 class-attention blocks, `num_classes 10`, `fc_params [(512,0.1)]`.
- Recipe (both arms, identical): ranger, flat+decay (weaver default), 50 epochs, batch 512, start-lr 1e-3, `--use-amp`, best = highest val accuracy. **Only** `--data-config` differs.
- Reproducibility: weaver has no `--seed` (D1); each arm repeated ___× for seed spread.

## Dry-run summary (from `/data/results/e1/smoke/`)
- Loss decreased, no AMP NaN: Arm P ___ / Arm S ___.
- Arm S make_weight distribution max/min ratio: ___ .
- Resume check (Arm P `--load-epoch 1`): ___ .
- Throughput: ___ entries/s → projected full-run wall time: Arm P ___ , Arm S ___ (sign-off if >7 GPU-days/arm).
- pred.root branch schema confirmed for eval_e1.py: ___ .

## Probe summary (from `/data/results/e1/probe/probe_report.json`)
- All 10 classes present, counts sane: ___ .
- (jet_pt, jet_sdmass) ranges → Arm S bins finalized: ___ (frac outside <0.5%: pt ___ , sd ___).

## Main results — test_20M (official)

| Metric | Arm P (ParT preproc) | Arm S (Sophon preproc) | Published ParT (ref) |
|---|---|---|---|
| Accuracy | ___ | ___ | 0.861 |
| Macro AUC (OvR) | ___ | ___ | **0.9877** |
| Mean rejection vs QCD @ ε=0.5 | ___ | ___ | — |
| Mean rejection vs QCD @ ε=0.3 | ___ | ___ | — |

### Per-class (AUC / Rej@0.5 / Rej@0.3 / TPR@1e-3 / TPR@1e-4), Arm P vs Arm S
*(from metrics_arm_p.json / metrics_arm_s.json)*

| Class | AUC P | AUC S | Rej0.3 P | Rej0.3 S |
|---|---|---|---|---|
| Hbb … Tbl | | | | |

## Preprocessing tax (paired bootstrap, from `tax.json`)
- Tax = Arm P − Arm S.
- ΔMacro AUC = ___ , 95% CI [___ , ___].
- Δlog10(mean rejection @0.3) = ___ , 95% CI [___ , ___].

## Gate verdict
- Arm P macro AUC = ___ ; gap to published 0.9877 = ___ .
- **VALIDATED / NOT VALIDATED** (threshold ±0.002).
- If not validated, diagnostics run (order per `EXTRACTED_GROUND_TRUTH.md` §8): ___ .

## Decisions & deviations log
- D1 no `--seed` in weaver → repeats for seed spread.
- D2 Arm S reweight ported to 10 JetClass-1 classes.
- D3 Arm S bins finalized from probe.
- D4 Arm S selection intentionally empty (no pT/mSD cut).
- D5 both arms `type: simple` + standard weaver eval.
- D6 weaver version: ___ (reinstalled? ___).
- D7 gate reference = published ParT 0.9877 (external line only).
- Other deviations: ___ .

## Deliverables checklist
- [ ] EXTRACTED_GROUND_TRUTH.md  [ ] environment.lock  [ ] data_arm_p.yaml + data_arm_s.yaml (final) + ParT_sophon_arch_10c.py
- [ ] probe_report.json (+ finalized Arm S bins)  [ ] make_weight artifacts + weight-dist plot  [ ] smoke logs
- [ ] full training logs/TensorBoard (both arms)  [ ] best checkpoints  [ ] pred.root (both arms)
- [ ] metrics_arm_p.json / metrics_arm_s.json / tax.json  [ ] this report filled in
