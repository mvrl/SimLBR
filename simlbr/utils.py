import os
import random

import kornia.augmentation as K
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data.dataloader import default_collate
from torchvision.transforms import v2


def set_seed(seed: int = 56) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def custom_collate(batch):
    keys = batch[0].keys()
    valid_batch = [
        data for data in batch
        if all(data[key] is not None for key in keys)
    ]
    if not valid_batch:
        raise ValueError("No valid samples in batch.")
    return default_collate(valid_batch)


def make_dinov3_transform(
    resize_size: int = 256,
    degradation_aug: bool = False,
    jpeg_quality: int | None = None,
    blur_sigma: float | None = None,
):
    to_tensor = v2.ToImage()
    resize = v2.Resize((resize_size, resize_size), antialias=True)
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.430, 0.411, 0.296),
        std=(0.213, 0.156, 0.143),
    )
    if not degradation_aug:
        return v2.Compose([to_tensor, resize, to_float, normalize])

    if jpeg_quality is None and blur_sigma is None:
        raise ValueError("jpeg_quality or blur_sigma must be provided for degradation_aug.")
    if jpeg_quality is not None and blur_sigma is not None:
        raise ValueError("Only one of jpeg_quality or blur_sigma should be provided.")

    if jpeg_quality is not None:
        print(f"Using JPEG perturbation with quality: {jpeg_quality}")
        perturbations = K.container.ImageSequential(
            K.RandomJPEG(jpeg_quality=(jpeg_quality, jpeg_quality), p=1.0)
        )
    else:
        print(f"Using Gaussian blur perturbation with sigma: {blur_sigma}")
        perturbations = K.container.ImageSequential(
            K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(blur_sigma, blur_sigma), p=1.0)
        )

    return v2.Compose([
        to_tensor,
        resize,
        to_float,
        v2.Lambda(lambda x: perturbations(x)[0]),
        normalize,
    ])


def cleanup():
    if torch.distributed.is_available() and dist.is_initialized():
        dist.destroy_process_group()
