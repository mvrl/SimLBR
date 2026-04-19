import glob
import os

import torch
from PIL import Image, ImageFile
from sklearn.utils import shuffle as sk_shuffle
from torch.utils.data import DataLoader, Dataset

from .utils import custom_collate, make_dinov3_transform

ImageFile.LOAD_TRUNCATED_IMAGES = True


def prepare_datasets(args):
    train_dataset = get_dataset(
        dataset_name=args.dataset_name,
        root_dir=args.data_dir,
        mode="train",
        model=args.train_model,
        degradation_aug=args.degradation_aug,
        jpeg_quality=args.jpeg_quality,
        blur_sigma=args.blur_sigma,
        fraction=1.0,
    )
    val_dataset = get_dataset(
        dataset_name=args.dataset_name,
        root_dir=args.data_dir,
        mode="test",
        model=args.val_model,
        fraction=args.ds_fraction,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=custom_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=custom_collate,
    )
    if args.test_model:
        test_dataset = get_dataset(
            dataset_name=args.dataset_name,
            root_dir=args.data_dir,
            mode="test",
            model=args.test_model,
            fraction=args.ds_fraction,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=custom_collate,
        )
    else:
        test_loader = None
    return train_loader, val_loader, test_loader


def get_dataset(
    dataset_name,
    root_dir,
    mode,
    model,
    degradation_aug=False,
    jpeg_quality=None,
    blur_sigma=None,
    fraction=1.0,
):
    if dataset_name == "aigc":
        if mode == "train" and model == "ProGAN":
            print("Loading ProGAN dataset for train")
            return ProGANDataset(
                root_dir=root_dir,
                mode=mode,
                model=model,
                degradation_aug=degradation_aug,
                jpeg_quality=jpeg_quality,
                blur_sigma=blur_sigma,
            )
        if mode == "train":
            raise ValueError("For AIGC, training is only allowed on ProGAN.")
        if model not in AIGCDataset.all_models and model != "combined":
            raise ValueError(f"Model {model} not available in AIGCDataset.")
        print(f"Loading {model} dataset for test")
        return AIGCDataset(
            root_dir=root_dir,
            mode=mode,
            model=model,
            degradation_aug=degradation_aug,
            jpeg_quality=jpeg_quality,
            blur_sigma=blur_sigma,
            fraction=fraction,
        )

    if dataset_name == "genimage":
        if model not in GenImageDataset.all_models and model != "combined":
            raise ValueError(f"Model {model} not available in GenImageDataset.")
        print(f"Loading {model} dataset for {mode}")
        return GenImageDataset(
            root_dir=root_dir,
            mode=mode,
            model=model,
            degradation_aug=degradation_aug,
            jpeg_quality=jpeg_quality,
            blur_sigma=blur_sigma,
            fraction=fraction,
        )

    raise ValueError(f"Invalid dataset name: {dataset_name}")


class ProGANDataset(Dataset):
    all_models = ["ProGAN"]

    def __init__(
        self,
        root_dir="/projects/bdec/adhakal2/data/fake_data/AIGC/AIGCDetectionBenchMark",
        mode="train",
        model="ProGAN",
        degradation_aug=False,
        jpeg_quality=None,
        blur_sigma=None,
    ):
        root_dir = os.path.join(root_dir, mode, model)
        self.all_files = glob.glob(os.path.join(root_dir, "*/*/*.png"))
        labels = []
        for image_path in self.all_files:
            parent = image_path.split("/")[-2]
            if parent == "0_real":
                labels.append(torch.tensor(0))
            elif parent == "1_fake":
                labels.append(torch.tensor(1))
            else:
                raise ValueError(f"Invalid parent folder name: {parent}")
        self.labels = torch.stack(labels)
        self.real_indices = torch.where(self.labels == 0)[0]
        if len(self.real_indices) == 0:
            raise ValueError("Dataset must contain at least one real sample.")
        self.transforms = make_dinov3_transform(
            degradation_aug=degradation_aug,
            jpeg_quality=jpeg_quality,
            blur_sigma=blur_sigma,
        )

    def __len__(self):
        return len(self.all_files)

    def _sample_real_index(self):
        return int(self.real_indices[torch.randint(len(self.real_indices), (1,))])

    def __getitem__(self, index):
        image_path = self.all_files[index]
        label = self.labels[index]
        anchor = Image.open(image_path).convert("RGB")
        anchor = self.transforms(anchor)
        if label.item() == 0:
            pair = torch.zeros_like(anchor)
        else:
            pair_path = self.all_files[self._sample_real_index()]
            pair = Image.open(pair_path).convert("RGB")
            pair = self.transforms(pair)
        return {"anchor": anchor, "pair": pair, "label": label}


class AIGCDataset(Dataset):
    all_models = [
        "ADM",
        "biggan",
        "cyclegan",
        "DALLE2",
        "gaugan",
        "Glide",
        "Midjourney",
        "progan",
        "stable_diffusion_v_1_4",
        "stable_diffusion_v_1_5",
        "stargan",
        "stylegan",
        "stylegan2",
        "VQDM",
        "whichfaceisreal",
        "wukong",
    ]

    def __init__(
        self,
        root_dir="/projects/bdec/adhakal2/data/fake_data/AIGC/AIGCDetectionBenchMark",
        model="ADM",
        mode="test",
        degradation_aug=False,
        jpeg_quality=None,
        blur_sigma=None,
        fraction=1.0,
    ):
        if model == "combined":
            all_models = self.all_models
        else:
            all_models = [model]

        self.all_files = []
        for model_name in all_models:
            model_path = os.path.join(root_dir, mode, model_name)
            if not os.path.exists(model_path):
                continue
            if set(os.listdir(model_path)) == {"0_real", "1_fake"}:
                self.all_files.extend(self._collect_real_fake(model_path, fraction))
            else:
                for folder_name in os.listdir(model_path):
                    folder_path = os.path.join(model_path, folder_name)
                    if set(os.listdir(folder_path)) != {"0_real", "1_fake"}:
                        raise ValueError(f"Invalid AIGC folder layout: {folder_path}")
                    self.all_files.extend(self._collect_real_fake(folder_path, fraction))

        self.real_files = [
            file_path for file_path in self.all_files
            if file_path.split("/")[-2] == "0_real"
        ]
        if len(self.real_files) == 0:
            raise ValueError("Dataset must contain at least one real sample.")
        self.transforms = make_dinov3_transform(
            degradation_aug=degradation_aug,
            jpeg_quality=jpeg_quality,
            blur_sigma=blur_sigma,
        )

    @staticmethod
    def _limit(paths, fraction):
        if fraction is not None and fraction < 1.0:
            return sk_shuffle(paths, random_state=42)[:int(len(paths) * fraction)]
        return paths

    def _collect_real_fake(self, parent, fraction):
        real_parent = os.path.join(parent, "0_real")
        fake_parent = os.path.join(parent, "1_fake")
        real_paths = [
            os.path.join(real_parent, img)
            for img in os.listdir(real_parent)
        ]
        fake_paths = [
            os.path.join(fake_parent, img)
            for img in os.listdir(fake_parent)
        ]
        return self._limit(real_paths, fraction) + self._limit(fake_paths, fraction)

    def __len__(self):
        return len(self.all_files)

    def _sample_real_path(self):
        real_idx = torch.randint(len(self.real_files), (1,)).item()
        return self.real_files[real_idx]

    def __getitem__(self, index):
        image_path = self.all_files[index]
        parent = image_path.split("/")[-2]
        if parent == "0_real":
            label = torch.tensor(0)
        elif parent == "1_fake":
            label = torch.tensor(1)
        else:
            raise ValueError(f"Invalid parent folder name: {parent}")

        anchor = Image.open(image_path).convert("RGB")
        anchor = self.transforms(anchor)
        if label.item() == 0:
            pair = torch.zeros_like(anchor)
        else:
            pair_path = self._sample_real_path()
            pair = Image.open(pair_path).convert("RGB")
            pair = self.transforms(pair)
        return {"anchor": anchor, "pair": pair, "label": label}


class GenImageDataset(Dataset):
    all_models = [
        "ADM",
        "BigGAN",
        "glide",
        "Midjourney",
        "stable_diffusion_v_1_4",
        "stable_diffusion_v_1_5",
        "VQDM",
        "wukong",
    ]

    def __init__(
        self,
        root_dir="/projects/bdec/adhakal2/data/fake_data/GenImage",
        model="BigGAN",
        mode="val",
        degradation_aug=False,
        jpeg_quality=None,
        blur_sigma=None,
        fraction=1.0,
    ):
        if mode == "test":
            mode = "val"
        all_models = self.all_models if model == "combined" else [model]

        self.real_files = []
        self.fake_files = []
        for model_name in all_models:
            real_paths = glob.glob(os.path.join(root_dir, model_name, f"*/{mode}/nature/*.JPEG"))
            fake_paths = glob.glob(os.path.join(root_dir, model_name, f"*/{mode}/ai/*.png"))
            fake_paths.extend(glob.glob(os.path.join(root_dir, model_name, f"*/{mode}/ai/*.PNG")))
            self.real_files.extend(AIGCDataset._limit(real_paths, fraction))
            self.fake_files.extend(AIGCDataset._limit(fake_paths, fraction))

        self.all_files = self.real_files + self.fake_files
        self.all_labels = [torch.tensor(0)] * len(self.real_files) + [torch.tensor(1)] * len(self.fake_files)
        if len(self.real_files) == 0:
            raise ValueError("Dataset must contain at least one real sample.")
        self.transforms = make_dinov3_transform(
            degradation_aug=degradation_aug,
            jpeg_quality=jpeg_quality,
            blur_sigma=blur_sigma,
        )

    def __len__(self):
        return len(self.all_files)

    def _sample_real_path(self):
        real_idx = torch.randint(len(self.real_files), (1,)).item()
        return self.real_files[real_idx]

    def __getitem__(self, index):
        image_path = self.all_files[index]
        label = self.all_labels[index]
        try:
            anchor = Image.open(image_path).convert("RGB")
            anchor = self.transforms(anchor)
            if label.item() == 0:
                pair = torch.zeros_like(anchor)
            else:
                pair_path = self._sample_real_path()
                pair = Image.open(pair_path).convert("RGB")
                pair = self.transforms(pair)
        except Exception as exc:
            print(f"Error loading image {image_path}: {exc}")
            anchor = None
            pair = None
        return {"anchor": anchor, "pair": pair, "label": label}

