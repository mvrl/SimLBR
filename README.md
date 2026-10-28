# (CVPR 2026) SimLBR: Learning to Detect Fake Images by Learning to Detect Real Images 🤖🖼️🤖
<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2602.20412-b31b1b.svg?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2602.20412)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Models-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/MVRL/SimLBR)

[Aayush Dhakal*](https://sites.wustl.edu/aayush/)&nbsp;&nbsp;&nbsp;
[Subash Khanal](https://subash-khanal.github.io/)&nbsp;&nbsp;&nbsp;
[Srikumar Sastry](https://vishu26.github.io/)&nbsp;&nbsp;&nbsp;
[Jacob Arndt](https://www.ornl.gov/staff-profile/jacob-w-arndt)&nbsp;&nbsp;&nbsp;
[Philipe Ambrozio Dias](https://www.ornl.gov/staff-profile/philipe-ambrozio-dias)&nbsp;&nbsp;&nbsp;
[Dalton Lunga](https://www.ornl.gov/staff-profile/dalton-d-lunga)&nbsp;&nbsp;&nbsp;
[Nathan Jacobs](https://jacobsn.github.io/)

</div>
<br>
This repo contains the code for the CVPR 2026 paper SimLBR. We introduce a new regularization objective, Latent Blending Regularization (LBR), for generalizable AI-generated Image Detection.
<br>
<br>
A frozen DINOv3 backbone extracts one CLS embedding per image, and a lightweight MLP head learns the real/fake classifier. When `--lbr` flag is enabled, real-image tokens are shifted asymmetrically towards fake-image tokens during training, creating pseudo-fake samples near the real distribution. Validation and test always use unmodified images.


## ⚙️ Setup


DINOv3 loading expects `DINO_V3_KEY`, the API key for dinov3 model:

```bash
export DINO_V3_KEY=API_KEY_FOR_DINO_V3_L16
```

Run commands from the repository root:

```bash
cd ./SimLBR
```

## 🤗 Pretrained Models

Pretrained detectors are hosted on Hugging Face at [MVRL/SimLBR](https://huggingface.co/MVRL/SimLBR). Each release file contains only the MLP head (~2.6 MB); the frozen DINOv3 ViT-L/16 backbone is loaded from Meta using `DINO_V3_KEY` as above.

| Variant | Training data | Benchmark | Avg. Acc |
|---|---|---|---|
| `aigc` | ProGAN | AIGCDetectionBenchMark (16 subsets) | 88.4 |
| `genimage` | GenImage SD v1.4 | GenImage (8 subsets) | 91.1* |

\* Corrected GenImage numbers; see the [ADM erratum](#erratum-genimage-adm-evaluation) below.

### Loading from Hugging Face

Install the Hugging Face dependencies:

```bash
pip install huggingface_hub safetensors
```

`SimLBR.from_pretrained` downloads the head from `MVRL/SimLBR` (cached in `~/.cache/huggingface`), builds the DINOv3 backbone, and returns the model in eval mode:

```python
import torch
from PIL import Image

from simlbr import SimLBR
from simlbr.utils import make_dinov3_transform

model = SimLBR.from_pretrained("aigc")  # or "genimage"
transform = make_dinov3_transform()     # 256x256 resize + the normalization the heads were trained with

image = transform(Image.open("path/to/image.png").convert("RGB")).unsqueeze(0)
with torch.no_grad():
    p_fake = torch.sigmoid(model(image)).item()  # > 0.5 means fake
```

Score images from the command line:

```bash
python -m simlbr.predict --pretrained aigc path/to/image1.png path/to/image2.jpg
```

### Erratum: GenImage ADM evaluation

Our original GenImage evaluation of the ADM subset contained **only real images**. GenImage's ADM fake images use an uppercase `.PNG` extension, and the embedding-extraction code only matched `*.png`, so all 6,000 ADM fakes were silently skipped. The reported ADM accuracy was therefore just the real-image accuracy.

Corrected results for the released `genimage` model (all other GenImage subsets, and all AIGCDetectionBenchMark results, are unaffected):

| | ADM Acc | ADM Real Acc | 
|---|---|---|
| Originally reported | 97.0 | 97.0 | 
| Corrected | 69.4 | 97.0 |

The released code loads both latest commit reproduces the corrected numbers. The training data of the released models did not include ADM, so the models themselves are unchanged.

## 🗄️ Data Layout

AIGC training uses `ProGAN`:

```text
AIGCDetectionBenchMark/
  train/ProGAN/<category>/0_real/*.png
  train/ProGAN/<category>/1_fake/*.png
  test/<model>/0_real/*
  test/<model>/1_fake/*
```

GenImage training uses `stable_diffusion_v_1_4`:

```text
GenImage/
  <model>/<category>/train/nature/*.JPEG
  <model>/<category>/train/ai/*.png
  <model>/<category>/val/nature/*.JPEG
  <model>/<category>/val/ai/*.png
```

Fake samples are labeled `1`; real samples are labeled `0`. During training, each fake anchor is paired with a random real image from the same dataset.

## 🏋️ Training

Baseline detector without latent blending:

```bash
python -m simlbr.train \
  --dataset_name aigc \
  --data_dir ../data/fake_data/AIGC/AIGCDetectionBenchMark \
  --train_model ProGAN \
  --val_model combined \
  --ds_fraction 0.05 \
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
  --data_dir ../data/fake_data/AIGC/AIGCDetectionBenchMark \
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

## 🕵️ Evaluation

Evaluate all subsets for a dataset:

```bash
python -m simlbr.evaluate \
  --dataset_name aigc \
  --data_dir ../data/fake_data/AIGC/AIGCDetectionBenchMark \
  --ckpt_path /path/to/checkpoint.ckpt \
  --devices 1
```

Evaluate selected subsets:

```bash
python -m simlbr.evaluate \
  --dataset_name aigc \
  --data_dir ../data/fake_data/AIGC/AIGCDetectionBenchMark \
  --ckpt_path /path/to/checkpoint.ckpt \
  --eval_datasets DALLE2 Midjourney
```

Evaluate a released pretrained model instead of a local checkpoint:

```bash
python -m simlbr.evaluate \
  --dataset_name aigc \
  --data_dir ../data/fake_data/AIGC/AIGCDetectionBenchMark \
  --pretrained aigc
```

Evaluation writes `evaluation_results.csv` next to the checkpoint run directory, or to `eval_results/pretrained_<variant>_on_<dataset>/` when using `--pretrained`. If `--eval_datasets` is not passed, this script launches evaluation across all generative models in the given dataset. 

## 🎌 Important Flags

- `--lbr`: enable Latent Blending Regularization (LBR) during training.
- `--lbrdist LOW HIGH`: alpha range for latent blending; default is `0.5 0.8`.
- `--hidden_layers`: number of MLP hidden layers after the DINOv3 CLS token.
- `--activation`: `relu` or `gelu`.
- `--dropout`: dropout inside the classifier MLP.
- `--wandb_mode`: use `online`, `offline`, or `disabled`.

## 🔖 Citation
```bibtex
@inproceedings{dhakal2026simlbr,
  title={SimLBR: Learning to Detect Fake Images by Learning to Detect Real Images},
  author={Dhakal, Aayush and Khanal, Subash and Sastry, Srikumar and Arndt, Jacob and Ambrozio Dias, Philipe and Lunga, Dalton and Jacobs, Nathan},
  booktitle={Computer Vision and Pattern Recognition},
  year={2026},
  organization={IEEE/CVF}
}
```
