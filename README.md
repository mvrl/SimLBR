# SimLBR

SimLBR is a compact CLS-token deepfake detector with optional latent blending regularization. A frozen DINOv3 backbone extracts one CLS embedding per image, and a lightweight MLP head learns the real/fake classifier. When `--lbr` is enabled, fake-image CLS tokens are blended toward paired real-image CLS tokens during training, creating pseudo-fake samples near the real distribution while validation and test always use unmodified images.

This repository is intentionally CLS-only. It does not include patch classifiers, patch-level LBR, intermediate-layer blending, attention reducers, or dual-branch fusion.

## Setup

Use the existing `mvrl` conda environment:

```bash
conda activate mvrl
```

DINOv3 loading expects `DINO_V3_KEY` to point to the local or remote DINOv3 weights used by `torch.hub.load`:

```bash
export DINO_V3_KEY=/path/to/dinov3_vitl16_weights.pth
```

Run commands from the repository root:

```bash
cd /projects/bdec/adhakal2/SimLBR
```

## Data Layout

AIGC training uses ProGAN:

```text
AIGCDetectionBenchMark/
  train/ProGAN/<category>/0_real/*.png
  train/ProGAN/<category>/1_fake/*.png
  test/<model>/0_real/*
  test/<model>/1_fake/*
```

GenImage uses:

```text
GenImage/
  <model>/<category>/train/nature/*.JPEG
  <model>/<category>/train/ai/*.png
  <model>/<category>/val/nature/*.JPEG
  <model>/<category>/val/ai/*.png
```

Fake samples are labeled `1`; real samples are labeled `0`. During training, each fake anchor is paired with a random real image from the same dataset.

## Training

Baseline CLS detector without latent blending:

```bash
python -m simlbr.train \
  --dataset_name aigc \
  --data_dir /projects/bdec/adhakal2/data/fake_data/AIGC/AIGCDetectionBenchMark \
  --train_model ProGAN \
  --val_model combined \
  --ds_fraction 0.2 \
  --batch_size 200 \
  --num_workers 20 \
  --max_epochs 5 \
  --devices 4 \
  --run_name aigc_cls_baseline
```

SimLBR training with CLS-token latent blending:

```bash
python -m simlbr.train \
  --dataset_name aigc \
  --data_dir /projects/bdec/adhakal2/data/fake_data/AIGC/AIGCDetectionBenchMark \
  --train_model ProGAN \
  --val_model combined \
  --ds_fraction 0.2 \
  --batch_size 200 \
  --num_workers 20 \
  --max_epochs 5 \
  --devices 4 \
  --lbr \
  --lbrdist 0.5 0.8 \
  --run_name aigc_simlbr
```

Fast development check:

```bash
python -m simlbr.train --fast_dev_run --wandb_mode disabled --accelerator cpu --devices 1
```

## Evaluation

Evaluate all subsets for a dataset:

```bash
python -m simlbr.evaluate \
  --dataset_name aigc \
  --data_dir /projects/bdec/adhakal2/data/fake_data/AIGC/AIGCDetectionBenchMark \
  --ckpt_path /path/to/checkpoint.ckpt \
  --devices 1
```

Evaluate selected subsets:

```bash
python -m simlbr.evaluate \
  --dataset_name aigc \
  --data_dir /projects/bdec/adhakal2/data/fake_data/AIGC/AIGCDetectionBenchMark \
  --ckpt_path /path/to/checkpoint.ckpt \
  --eval_datasets DALLE2 Midjourney
```

Evaluation writes `evaluation_results.csv` next to the checkpoint run directory.

## Important Flags

- `--lbr`: enable CLS-token latent blending during training.
- `--lbrdist LOW HIGH`: alpha range for latent blending; default is `0.5 0.8`.
- `--hidden_layers`: number of MLP hidden layers after the DINOv3 CLS token.
- `--activation`: `relu` or `gelu`.
- `--dropout`: dropout inside the classifier MLP.
- `--wandb_mode`: use `online`, `offline`, or `disabled`.

Patch-level LLBR flags such as `--patch_reduction`, `--cls_selection`, and `--lbr_blend_layer` are intentionally not part of this repo.
