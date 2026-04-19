import argparse


def add_dataset_args(parser):
    group = parser.add_argument_group("Dataset and Dataloader Arguments")
    group.add_argument("--dataset_name", type=str, choices=["aigc", "genimage"], default="aigc")
    group.add_argument("--data_dir", type=str, default="../data/fake_data/AIGC/AIGCDetectionBenchMark")
    group.add_argument("--train_model", type=str, default="ProGAN")
    group.add_argument("--val_model", type=str, default="DALLE2")
    group.add_argument("--test_model", type=str, default="")
    group.add_argument("--degradation_aug", action="store_true")
    group.add_argument("--jpeg_quality", type=int, default=None)
    group.add_argument("--blur_sigma", type=float, default=None)
    group.add_argument("--ds_fraction", type=float, default=0.05)
    group.add_argument("--batch_size", type=int, default=32)
    group.add_argument("--num_workers", type=int, default=4)
    return parser


def add_trainer_args(parser):
    group = parser.add_argument_group("Lightning Trainer Arguments")
    group.add_argument("--max_epochs", type=int, default=10)
    group.add_argument("--devices", type=int, default=1)
    group.add_argument("--precision", type=str, default="16-mixed")
    group.add_argument("--accelerator", type=str, default="gpu")
    group.add_argument("--fast_dev_run", action="store_true")
    group.add_argument("--val_check_interval", type=float, default=1.0)
    return parser


def add_logging_args(parser):
    group = parser.add_argument_group("Logging Arguments")
    group.add_argument("--log_dir", type=str, default="../logs")
    group.add_argument("--run_name", type=str, default="test_run")
    group.add_argument("--project_name", type=str, default="SimLBR")
    group.add_argument("--wandb_mode", type=str, default="online")
    group.add_argument("--ckpt_path", type=str, default=None)
    return parser


def add_model_args(parser):
    group = parser.add_argument_group("Model Arguments")
    group.add_argument("--backbone", type=str, choices=["dinov3"], default="dinov3")
    group.add_argument("--lr", type=float, default=1e-4)
    group.add_argument("--wt_decay", type=float, default=1e-2)
    group.add_argument("--activation", type=str, choices=["relu", "gelu"], default="relu")
    group.add_argument("--hidden_layers", type=int, default=2)
    group.add_argument("--dropout", type=float, default=0.3)
    group.add_argument("--lbr", action="store_true")
    group.add_argument("--lbrdist", type=float, nargs=2, default=(0.5, 0.8))
    return parser


def get_args(argv=None):
    parser = argparse.ArgumentParser(description="Train a CLS-token SimLBR detector.")
    add_dataset_args(parser)
    add_trainer_args(parser)
    add_logging_args(parser)
    add_model_args(parser)
    return parser.parse_args(argv)
