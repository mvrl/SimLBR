import os

import torch
import torch.nn as nn


def get_dinov3():
    print("Using DINOv3 as backbone")
    model_url = os.environ["DINO_V3_KEY"]
    model = torch.hub.load("facebookresearch/dinov3", "dinov3_vitl16", weights=model_url)
    feat_dim = 1024
    return model, feat_dim


class VisionBackbone(nn.Module):
    def __init__(self, model_name: str = "dinov3"):
        super().__init__()
        if model_name != "dinov3":
            raise ValueError(f"Unsupported backbone: {model_name}. SimLBR only supports dinov3.")

        self.model_name = model_name
        self.model, self.feat_dim = get_dinov3()

    def forward(self, x):
        features = self.model.forward_features(x)
        return {
            "cls_token": features["x_norm_clstoken"],
        }
