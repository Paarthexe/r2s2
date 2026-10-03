#!/usr/bin/env python3
import argparse
import sys
import os

from data_prep.config import DatasetConfig
from data_prep.generator import DatasetGenerator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate synthetic conversational speech separation dataset from LibriSpeech."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dataset_config.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory to save mixtures and metadata.",
    )
    parser.add_argument(
        "--max-runtime-hours",
        type=float,
        default=None,
        help="Total wall-clock runtime allocation in hours across splits.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of successful samples to generate per split (optional).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of parallel worker processes.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Batch chunk size for parallel joblib dispatch.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Dataset splits to generate (default: train val test).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if os.path.exists(args.config):
        config = DatasetConfig.from_yaml(args.config)
    else:
        print(f"Config file '{args.config}' not found, using default configuration.")
        config = DatasetConfig()

    # CLI Overrides
    if args.output_dir:
        config.execution.output_dir = args.output_dir
    if args.max_runtime_hours is not None:
        config.execution.max_runtime_hours = args.max_runtime_hours
    if args.num_workers is not None:
        config.execution.num_workers = args.num_workers
    if args.chunk_size is not None:
        config.execution.chunk_size = args.chunk_size
    if args.seed is not None:
        config.execution.seed = args.seed

    print("=" * 60)
    print("Speech Separation Dataset Preparation")
    print(f"Output Directory  : {config.execution.output_dir}")
    print(f"Workers           : {config.execution.num_workers}")
    print(f"Chunk Size        : {config.execution.chunk_size}")
    print(f"Max Runtime (hrs) : {config.execution.max_runtime_hours}")
    print(f"Splits            : {args.splits}")
    print(f"Max Samples/Split : {args.max_samples if args.max_samples else 'Unbounded (time-based)'}")
    print("=" * 60)

    generator = DatasetGenerator(config)
    generator.run(
        splits=args.splits,
        max_runtime_hours=args.max_runtime_hours,
        max_samples_per_split=args.max_samples,
    )


if __name__ == "__main__":
    main()
