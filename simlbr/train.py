import os

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from .config import get_args
from .datasets import prepare_datasets
from .model import SimLBR
from .utils import cleanup, set_seed


def main(args):
    train_loader, val_loader, _ = prepare_datasets(args)
    if args.ckpt_path:
        print(f"Loading model from checkpoint: {args.ckpt_path}")
        model = SimLBR.load_from_checkpoint(args.ckpt_path)
    else:
        model = SimLBR(
            backbone=args.backbone,
            lr=args.lr,
            wt_decay=args.wt_decay,
            activation=args.activation,
            hidden_layers=args.hidden_layers,
            dropout=args.dropout,
            lbr=args.lbr,
            lbrdist=args.lbrdist,
        )

    wb_logger = WandbLogger(
        project=args.project_name,
        name=args.run_name,
        save_dir=args.log_dir,
        mode=args.wandb_mode,
    )
    ckpt_dir = os.path.join(args.log_dir, args.project_name, args.run_name, "checkpoints")
    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="{epoch:02d}-{step:02d}-{val_acc:.3f}",
            monitor="val_acc",
            mode="max",
            save_top_k=1,
            save_last=True,
        ),
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="{epoch:02d}-{step:02d}-{val_loss:.3f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        val_check_interval=args.val_check_interval,
        log_every_n_steps=10,
        limit_val_batches=1.0,
        num_sanity_val_steps=0,
        strategy="ddp_find_unused_parameters_false",
        logger=wb_logger,
        callbacks=callbacks,
        fast_dev_run=args.fast_dev_run,
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    args = get_args()
    set_seed(56)
    torch.set_float32_matmul_precision("medium")
    main(args)
    cleanup()
