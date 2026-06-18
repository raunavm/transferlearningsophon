# Transfer learning with Sophon on JetClass-I

Scaling-law comparison of three adaptation strategies (frozen + MLP, partial fine-tune, full fine-tune) for the [Sophon](https://github.com/jet-universe/sophon) foundation model on the 10-class [JetClass-I](https://zenodo.org/records/6619768) benchmark.

## Setup

```sh
conda create -n sophon python=3.10 -y && conda activate sophon
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -e .

mkdir -p models/JetClassII_Sophon
curl -fSL -o models/JetClassII_Sophon/model.pt \
  "https://huggingface.co/jet-universe/sophon/resolve/main/models/JetClassII_Sophon/model.pt"
```

## 1. Extract embeddings

```sh
python3 inference_all_classes.py \
  --checkpoint models/JetClassII_Sophon/model.pt \
  --root-dir data/train_100M \
  --events-per-class 0 \
  --output-dir embeddings_pretrained_full --format npy
```

## 2. Train

```sh
# Frozen + MLP sweep
python3 scripts/train_frozen_sweep.py \
  --emb-dir embeddings_pretrained_full \
  --output-dir results/frozen_base \
  --sizes 10000,30000,100000,300000,1000000,3000000,10000000,30000000,100000000 \
  --seeds 42,123,456

# Partial / full fine-tune sweep
python3 scripts/train_finetune_sweep.py \
  --train-dir /data/features/train_100M \
  --val-dir   /data/features/val_5M \
  --test-dir  /data/features/test_20M \
  --strategy partial_ft \
  --checkpoint models/JetClassII_Sophon/model.pt \
  --sizes 10000,30000,100000,300000,1000000,3000000,10000000 \
  --seeds 42,123,456 \
  --output-dir results/partial_ft \
  --materialize-train --skip-existing

python3 scripts/collect_results.py   # writes results/sweep_results.csv
```

## 3. Figures

```sh
python3 scripts/plot_scaling.py
python3 scripts/plot_umap.py --pretrained-dir embeddings_pretrained_full --finetuned-dir <ft_embeddings>
python3 scripts/plot_roc_3strategies.py \
  --frozen-mlp   results/frozen_base/frozen_base_10000000_42/best_model.pt \
  --partial-ckpt results/partial_ft/partial_ft_10000000_42/best_model.pt \
  --full-ckpt    results/full_ft/full_ft_10000000_42/best_model.pt \
  --pretrained   models/JetClassII_Sophon/model.pt \
  --features-dir /data/features/test_20M \
  --embeddings-dir embeddings_pretrained_full
```

## ParT comparison

The public supervised Particle Transformer is included as a reference. Extracts the 128-d CLS latent on JetClass test jets, sanity-checks against the published 0.988 macro AUC, and renders a UMAP / t-SNE / PCA 3-panel plus a 3-way UMAP comparison vs Sophon.

```sh
curl -fSL -o models/ParT/ParT_full.pt \
  "https://github.com/jet-universe/particle_transformer/raw/main/models/ParT_full.pt"

python3 scripts/extract_part_embeddings.py \
  --checkpoint models/ParT/ParT_full.pt \
  --root-dir /data/JetClass/Pythia/test_20M \
  --events-per-class 10000 \
  --output-dir embeddings_part_full_test

python3 scripts/part_sanity_check.py --emb-dir embeddings_part_full_test
python3 scripts/plot_part_reductions.py --emb-dir embeddings_part_full_test \
  --sophon-dir embeddings_test_20M \
  --sophon-ft-dir embeddings_ft_full_10M_seed42_test100k
```

## Layout

```
inference_all_classes.py        Sophon step-1 embedding extraction
src/{data,models,utils}/        package code (incl. Sophon + ParT wrappers)
plots/style.py                  figure style
scripts/                        training sweeps, ParT pipeline, figure renderers
k8s/                            kubernetes job templates
networks/                       reference Sophon ParT definition
results/sweep_results.csv       9 sizes x 3 strategies x seeds
results/main_plots/             poster figures (committed)
```


