import argparse
import os

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .datasets import AIGCDataset, GenImageDataset
from .model import SimLBR
from .utils import custom_collate


def get_args():
    parser = argparse.ArgumentParser(description="Evaluate a SimLBR checkpoint.")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--dataset_name", type=str, choices=["aigc", "genimage"], default="aigc")
    parser.add_argument("--data_dir", type=str, default="/projects/bdec/adhakal2/data/fake_data/AIGC/AIGCDetectionBenchMark")
    parser.add_argument("--eval_datasets", nargs="*", default=[])
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=28)
    return parser.parse_args()


def get_fake_dataset(args):
    if args.dataset_name == "aigc":
        return AIGCDataset
    if args.dataset_name == "genimage":
        return GenImageDataset
    raise ValueError(f"Unsupported dataset name: {args.dataset_name}")


def resolve_eval_datasets(args, fake_dataset):
    return args.eval_datasets if args.eval_datasets else fake_dataset.all_models


def evaluate_subset(args, trainer, model, fake_dataset, subset):
    print(f"Evaluating on subset: {subset}")
    test_dataset = fake_dataset(
        root_dir=args.data_dir,
        model=subset,
        mode="test",
        degradation_aug=False,
        jpeg_quality=None,
        blur_sigma=None,
        fraction=1.0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
        collate_fn=custom_collate,
    )
    test_results = trainer.test(model, dataloaders=test_loader)

    all_labels = model.all_labels.int()
    correct = model.correct.int()
    real_mask = all_labels == 0
    fake_mask = all_labels == 1
    real_acc = correct[real_mask].float().mean().item() if real_mask.any() else float("nan")
    fake_acc = correct[fake_mask].float().mean().item() if fake_mask.any() else float("nan")

    return {
        "subset": subset,
        "accuracy": test_results[0]["predict_acc"],
        "real_accuracy": real_acc,
        "fake_accuracy": fake_acc,
        "ap": test_results[0]["predict_ap"],
        "f1": test_results[0]["predict_f1"],
    }


def evaluate(args, trainer, model):
    fake_dataset = get_fake_dataset(args)
    rows = [
        evaluate_subset(args, trainer, model, fake_dataset, subset)
        for subset in resolve_eval_datasets(args, fake_dataset)
    ]
    df = pd.DataFrame(rows)
    df.loc[len(df)] = {
        "subset": "avg",
        "accuracy": np.mean(df["accuracy"]),
        "real_accuracy": np.mean(df["real_accuracy"]),
        "fake_accuracy": np.mean(df["fake_accuracy"]),
        "ap": np.mean(df["ap"]),
        "f1": np.mean(df["f1"]),
    }
    return df


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    args = get_args()
    model = SimLBR.load_from_checkpoint(args.ckpt_path)
    trainer = pl.Trainer(accelerator="gpu", devices=args.devices, precision="16-mixed")
    results = evaluate(args, trainer, model)
    eval_dir = os.path.join(os.path.dirname(os.path.dirname(args.ckpt_path)), "eval_results")
    os.makedirs(eval_dir, exist_ok=True)
    out_path = os.path.join(eval_dir, "evaluation_results.csv")
    results.to_csv(out_path, index=False)
    print(results)
    print(f"Saved evaluation results to {out_path}")
