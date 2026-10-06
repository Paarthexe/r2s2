#!/usr/bin/env python3
import os
import argparse

from inference.pipeline import load_model_for_inference, separate_audio_file
from models.config import ModelConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate / Run R2S2 recursive speech separation on audio.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best.pt",
        help="Path to trained model checkpoint (.pt).",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default="configs/model_config.json",
        help="Path to model configuration JSON.",
    )
    parser.add_argument(
        "--input-audio",
        type=str,
        required=True,
        help="Path to mixture audio file (.wav or .flac).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./separated_outputs",
        help="Directory to save separated speaker audio files.",
    )
    parser.add_argument(
        "--rms-threshold",
        type=float,
        default=0.01,
        help="Stopping energy threshold for residual mixture.",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=6,
        help="Maximum speaker extraction iterations.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Target sample rate in Hz.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run inference on ('cuda', 'cpu', 'mps').",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model_config = None
    if os.path.exists(args.model_config):
        model_config = ModelConfig.from_json(args.model_config)

    print(f"Loading checkpoint from: {args.checkpoint}")
    model = load_model_for_inference(
        checkpoint_path=args.checkpoint,
        config=model_config,
        device=args.device,
    )

    print(f"Separating audio: {args.input_audio}")
    output_files = separate_audio_file(
        model=model,
        input_path=args.input_audio,
        output_dir=args.output_dir,
        target_sr=args.sample_rate,
        rms_threshold=args.rms_threshold,
        max_iters=args.max_iters,
        device=args.device,
    )

    print(f"\nSeparation complete! Separated {len(output_files)} speakers:")
    for f in output_files:
        print(f"  - {f}")


if __name__ == "__main__":
    main()
