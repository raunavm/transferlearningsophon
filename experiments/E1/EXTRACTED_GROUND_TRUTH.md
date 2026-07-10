# E1 — Extracted Ground Truth (verification pass)

Every value below was read from the primary repos at pinned commits on **2026-07-09**. Citations are `file:line`. The right-hand column flags **DRIFT** vs the E1 agent prompt's §2 where the repos disagree with it.

## Pinned commits (record for reproducibility)

| Repo | Commit | Date | Branch |
|---|---|---|---|
| jet-universe/particle_transformer | `2925bdb249e8ef78560cc2b9b651eda3615da8c7` | 2024-05-13 | main |
| jet-universe/sophon | `9dd6dd6a261aa6d5fd2e56f015068127b36854f9` | 2024-08-16 | main |
| hqucms/weaver-core | `c97de3c83d2bc74d2b444d5afa24d4696e8be860` | 2024-10-23 | **dev/custom_train_eval** |

Local clones: `/Users/raunavmendiratta/e1_refs/{particle_transformer,sophon,weaver-core}`.

## 1. Optimizer — `ranger` (both arms)

`weaver-core/weaver/utils/nn/optimizer/ranger.py:5-11`:
```
Ranger = Lookahead( RAdam(lr, betas=(0.95,0.999), eps=1e-5, weight_decay=0), alpha=0.5, k=6 )
```
Matches the ParT paper (β=(0.95,0.999), ε=1e-5) and prompt §1 **exactly**. No drift. Wired at `train.py:510-512` (`Ranger(parameters, lr=args.start_lr, **optimizer_options)`).

## 2. LR schedule — `flat+decay` is the weaver **default** (`train.py:122`), not tied to ranger

`train.py:532-544` (`--lr-scheduler flat+decay`, the default, used because we do NOT override it):
```
num_decay_epochs = max(1, int(num_epochs * 0.3))       # 50 epochs -> 15
milestones       = range(num_epochs - num_decay_epochs, num_epochs)   # [35..49]
gamma            = 0.01 ** (1/num_decay_epochs)         # per-epoch factor -> reaches 0.01x start LR
scheduler        = MultiStepLR(opt, milestones, gamma)  # (no lr-mult groups for from-scratch)
```
= **LR constant for first 70% (35 epochs), then exponential per-epoch decay to 1% of start LR over the last 15.** This is the ParT-paper schedule. The paper's "every 20k iterations" ≈ per-epoch here (samples_per_epoch 10.24M ÷ batch 512 ≈ 20k steps/epoch). No drift; **do not pass `--lr-scheduler`** (default is correct).

## 3. Training recipe — official ParT-on-JetClass (both arms use this; it is the JetClass recipe)

`particle_transformer/train_JetClass.sh:25-33,61-91`:
- `epochs=50`
- `samples_per_epoch = 10000*1024/NGPUS` (= 10.24M / NGPUS)
- `samples_per_epoch_val = 10000*128` (= 1.28M)
- `--num-workers 2 --fetch-step 0.01`
- ParT model: `--batch-size 512 --start-lr 1e-3`, `--use-amp`, `--optimizer ranger`, `--gpus 0`
- best model = highest **validation accuracy** checkpoint (see §6); test once at end.

512M samples over 50 epochs ≈ **1M optimizer steps @ batch 512** (paper's "1M iterations ≈ 5 epochs" convention → 50 epochs = 10× that ×... note: weaver's "epoch" = `samples_per_epoch`, not a full pass over 100M; 50 weaver-epochs = 512M samples seen). No drift vs prompt §2a.

**DDP note** (`train_JetClass.sh:19-23`): `NGPUS>1` → `torchrun --nproc_per_node=$NGPUS weaver --backend nccl`; **scale `--start-lr` linearly with NGPUS** (Sophon example: 4 GPUs → lr 2e-3 from base 5e-4). For E1 keep base lr 1e-3 × NGPUS, identical across arms.

## 4. Two preprocessings — exact diff (the ONLY thing that differs between arms)

**Shared, byte-identical** in both YAMLs: `pf_points`=(part_deta,part_dphi); the 17 `pf_features` with identical manual standardization `[part_pt(*)_log,1.7,0.7], [part_e(*)_log,2.0,0.7], [part_logptrel,-4.7,0.7], [part_logerel,-4.7,0.7], [part_deltaR,0.2,4.0], part_charge, 5 PID flags, part_d0=tanh(d0val), [part_d0err,0,1,0,1], part_dz=tanh(dzval), [part_dzerr,0,1,0,1], part_deta, part_dphi`; `pf_mask`; all `length:128, pad_mode:wrap` (mask `constant`); `preprocess: {method: manual, data_fraction: 0.5}`; identical `observers` (incl. **`jet_sdmass`** and `jet_pt`).

**Arm P** = `particle_transformer/data/JetClass/JetClass_full.yaml` (use verbatim):
- `part_pt_log = np.log(part_pt)`, `part_e_log = np.log(part_energy)`; `part_pt = np.hypot(part_px, part_py)` (`:9-11`)
- `pf_vectors` = raw `(part_px, part_py, part_pz, part_energy)` (`:59-66`)
- `selection:` **EMPTY** (`:1-3`); `weights:` **EMPTY** (`:95`) — JetClass-1 is class-balanced, no reweighting
- `labels: type: simple, value: [label_QCD, label_Hbb, label_Hcc, label_Hgg, label_H4q, label_Hqql, label_Zqq, label_Wqq, label_Tbqq, label_Tbl]` (`:73-77`)

**Arm S** = port of `sophon/data/JetClassII/JetClassII_full.yaml`; the ONLY changes vs Arm P:
- scaled kinematics (`sophon yaml:12-20`): `part_{px,py,pz,energy}_scale = part_x / jet_pt * 500`; `part_pt_scale = hypot(px_scale,py_scale)`; `part_pt_scale_log = log(part_pt_scale)` (replaces part_pt_log, same 1.7/0.7); `part_e_scale_log = log(part_energy_scale)` (replaces part_e_log, same 2.0/0.7). `part_logptrel`/`part_logerel` are scale-invariant — unchanged.
- `pf_vectors` = the four `*_scale` vectors (`sophon yaml:105-112`).
- `weights:` flat reweighting on **(jet_pt, jet_sdmass)** (`sophon yaml:143-171`). **PORT (not verbatim):** Sophon reweights 30 `jet_label`-based meta-groups over 188 classes — those branches DO NOT EXIST in JetClass-1. Arm S reweights the **10 JetClass-1 label branches**, `reweight_method: flat`, `use_precomputed_weights: false`, uniform `class_weights` 0.1. **Bins must be re-derived from JetClass-1's real (jet_pt, jet_sdmass) ranges** (Sophon's 200–2500 / 20–500 are for its selection; JetClass-1 pT is ~500–1000 GeV and has light jets with mSD≈0 → start mSD at 0). ⇒ the probe job finalizes the bin edges so <0.5% of jets fall outside.
- `selection:` **EMPTY** — **stated design decision**: Sophon's `(200<pT<2500) & (20<mSD<500)` cut is dataset-specific; applying an mSD cut to JetClass-1 would silently change the evaluated population and contaminate the tax. Document this.
- Standardization constants unchanged — Sophon itself reuses ParT's (1.7/0.7 etc.) on the scaled vars.

## 5. Network — one file for BOTH arms (Sophon architecture)

`sophon/networks/example_ParticleTransformer_sophon.py:56-89` `get_model()`:
`input_dim=17, pair_input_dim=4, use_pre_activation_pair=True, embed_dims=[128,512,128], pair_embed_dims=[64,64,64], num_heads=8, num_layers=8, num_cls_layers=2, cls_block_params={dropout:0,...}, fc_params=[], activation=gelu, trim=True`. Set via CLI `-o num_classes 10 -o fc_params [(512,0.1)]`.
→ **8 particle-attention + 2 class-attention blocks** (corrects the paper's "6+2"). `get_loss` = plain `CrossEntropyLoss` (`:92-93`).

**Architecture delta vs plain ParT** (`particle_transformer/networks/example_ParticleTransformer.py` get_model): plain ParT has `use_pre_activation_pair=False` and `fc_params=[]`; everything else identical (8 layers, 2 cls layers, embed dims same). So "Sophon-arch + ParT-preproc" (Arm P) differs from official ParT by **{pre-activation pair = True, FC head (512,0.1)}**. If Arm P misses the 0.002 gate, first diagnostic = also train plain `example_ParticleTransformer.py` on Arm P's YAML.

### ⚠ LANDMINE — must fix before running at 10 classes
`example_ParticleTransformer_sophon.py` registers `get_train_fn`/`get_evaluate_fn` (`:96-101`) → custom eval `evaluate_classification_sophon` hardcodes 188-class indices: `scores[:, 161:188]` for QCD, `scores[:,0]`/`scores[:,1]` as Xbb/Xcc, `labels['truth_label']` (`:262-269`). At 10 classes this indexes out of range and assumes a `truth_label` custom label. **Fix:** ship `ParT_sophon_arch_10c.py` = the file with `get_train_fn`, `get_evaluate_fn`, and the two custom functions **deleted**, so weaver falls back to its standard `train_classification`/`evaluate_classification` (`weaver/utils/nn/tools.py`), which work with `labels: type: simple`. Use the SAME network file for both arms.

## 6. Best-epoch selection & prediction

`train.py:921-934`: classification → `is_best_epoch = valid_metric > best_valid_metric`; `valid_metric` = `evaluate(...)` return = `total_correct/count` = **validation accuracy**. Best saved to `{model_prefix}_best_epoch_state.pt`. `--predict` (`:800-801`, `:957`) loads `_best_epoch_state.pt` and runs on `--data-test`. Matches prompt. Do NOT evaluate test until best checkpoint is frozen.

## 7. DRIFT / decisions log (read carefully)

| # | Item | Finding | Action |
|---|---|---|---|
| D1 | **`--seed` flag** | weaver `dev/custom_train_eval` has **NO global seed** — only per-worker numpy seed from `worker_info.seed` (`dataset.py:166-167`). Model init + shuffle are unseeded, as when published ParT was produced. | Cannot set "seed 42/43." Estimate seed spread by **repeating each arm ≥2×** (independent runs). Document as faithful-to-published. |
| D2 | Arm S reweight classes | Sophon's 30 meta-groups use `jet_label` 0–187 (absent in JetClass-1). | Port to the 10 JetClass-1 label branches, uniform class_weights. |
| D3 | Arm S reweight bins | Sophon's 200–2500 / 20–500 don't fit JetClass-1. | Re-derive from data via the probe job; verify <0.5% out-of-range. |
| D4 | Arm S selection | Sophon applies a pT/mSD cut. | **Drop it** for JetClass-1 (would change evaluated population). Stated decision. |
| D5 | Label handling | Sophon uses `type: custom truth_label`; JetClass-1 is one-hot. | Both arms use `type: simple` (10 branches) + standard weaver eval. |
| D6 | weaver in container | Probe confirmed image has **weaver 0.4.17** (released PyPI), whose `ParticleTransformer` has **no `_forward_encoder`/`_forward_aggregator`** split API (that's the dev/custom_train_eval branch). The Sophon wrapper's split-forward crashed (`AttributeError`). | **Rewrote the network `forward` to the standard `self.mod(features, v=lorentz_vectors, mask=mask)`** (identical output; the plain-ParT wrapper pattern). No reinstall needed — 0.4.17 has standard weaver train/eval, `flat+decay`, and the full architecture. Faithful. |
| D8 | Arm S reweight overflow | Probe: `reweight_discard_under_overflow: True` is weaver's default, and jet_sdmass reaches ~497 while provisional bins stopped at 260 → 0.62% of jets (top/QCD tails) would be **silently discarded**. | Extended Arm S mass bins to 550 (pt padded 499/1001) so ~0% is discarded. Confirmed by probe. |
| D9 | `--auto-clean` not in image weaver | The image's **0.4.17** CLI has no `--auto-clean` (it exists only in the dev/custom_train_eval clone I ground-truthed). Arm S crashed at arg-parse (`unrecognized arguments: --auto-clean`). **Lesson: verify weaver CLI flags against the image's 0.4.17 usage, not the dev clone.** | Removed `--auto-clean`; 0.4.17 keeps all per-epoch checkpoints (~1.35 GB/arm, fine vs free space; also better for best-epoch selection). `--load-epoch` and `--no-remake-weights` ARE in 0.4.17 (confirmed from its usage). |
| D7 | Gate reference number | ParT paper Table 1 (arXiv:2202.03772), JetClass all-classes: **Accuracy 0.861, macro AUC 0.9877**. README has no inline table; cite the paper. | Gate = Arm P macro AUC within ~0.002 of **0.9877** on official test_20M. External reference line only — never a training target. |

## 8. Gate (restated)

Arm P (Sophon-arch, ParT preprocessing, weaver, 50 ep, ranger, flat+decay, amp) trained from scratch on JetClass-1 → **macro AUC within ~0.002 of 0.9877** on the official 20M test set ⇒ pipeline VALIDATED. Else diagnose in order: (1) plain-ParT-network control, (2) LR-schedule shape from logs, (3) AMP off short control, (4) data integrity/class balance, (5) effective samples seen vs 512M, (6) weaver drift. Never fix a gap by changing evaluation.
