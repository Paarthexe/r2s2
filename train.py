#!/usr/bin/env python3
import os
import argparse

from training.config import TrainConfig
from training.engine import train


def parse_args():
    parser = argparse.ArgumentParser(description="Train R2S2 Recursive Residual Speech Separation model.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.json",
        help="Path to JSON training configuration file.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Path to dataset directory containing train/ and val/ splits.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Total training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Training batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory to save model checkpoints.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device ('cuda', 'cpu', 'mps').",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of DataLoader worker processes.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if os.path.exists(args.config):
        config = TrainConfig.from_json(args.config)
    else:
        print(f"Config '{args.config}' not found, using default TrainConfig.")
        config = TrainConfig()

    # CLI Overrides
    if args.dataset_dir:
        config.dataset_dir = args.dataset_dir
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.lr = args.lr
    if args.checkpoint_dir:
        config.checkpoint_dir = args.checkpoint_dir
    if args.resume:
        config.resume_checkpoint = args.resume
    if args.device:
        config.device = args.device
    if args.num_workers is not None:
        config.num_workers = args.num_workers

    train(config)


if __name__ == "__main__":
    main()
