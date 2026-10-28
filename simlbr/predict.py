import argparse

import torch
from PIL import Image

from .model import PRETRAINED_VARIANTS, SimLBR
from .utils import make_dinov3_transform


def get_args():
    parser = argparse.ArgumentParser(description="Score images with a pretrained SimLBR detector.")
    parser.add_argument("images", nargs="+", help="Image file paths.")
    parser.add_argument("--pretrained", type=str, choices=PRETRAINED_VARIANTS, default="aigc")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.no_grad()
def predict(model, image_paths, device):
    transform = make_dinov3_transform()
    images = torch.stack([transform(Image.open(path).convert("RGB")) for path in image_paths])
    logits = model(images.to(device)).squeeze(1)
    return torch.sigmoid(logits).cpu()


if __name__ == "__main__":
    args = get_args()
    model = SimLBR.from_pretrained(args.pretrained).to(args.device)
    probs = predict(model, args.images, args.device)
    for path, prob in zip(args.images, probs.tolist()):
        label = "fake" if prob > args.threshold else "real"
        print(f"{path}\tp(fake)={prob:.4f}\t{label}")
