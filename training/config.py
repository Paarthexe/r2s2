import os
import json
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
import torch

from models.config import ModelConfig


@dataclass
class TrainConfig:
    """Hyperparameters for R2S2 training pipeline."""

    dataset_dir: str = "./conversational_dataset_v2"
    sample_rate: int = 16000
    crop_seconds: float = 4.0
    batch_size: int = 16
    epochs: int = 40
    lr: float = 1e-3
    weight_decay: float = 1e-2
    warmup_steps: int = 400
    clip_grad: float = 5.0
    stft_lambda_max: float = 0.1
    stft_lambda_floor: float = 0.01
    stft_warmup_start: int = 8
    stft_warmup_end: int = 15
    unroll_prob: float = 0.5
    unroll_start_epoch: int = 20
    consistency_weight: float = 0.1
    val_interval: int = 5
    num_workers: int = 4
    device: Optional[str] = None
    checkpoint_dir: str = "./checkpoints"
    resume_checkpoint: Optional[str] = None
    model: ModelConfig = field(default_factory=ModelConfig)

    @property
    def crop_samples(self) -> int:
        return int(self.sample_rate * self.crop_seconds)

    @property
    def resolved_device(self) -> str:
        if self.device:
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def from_json(cls, path: str) -> "TrainConfig":
        """Load configuration from a JSON file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file not found: {path}")
        with open(path, "r") as f:
            data = json.load(f)

        model_data = data.pop("model", {})
        if isinstance(model_data, dict):
            model_cfg = ModelConfig(**{k: v for k, v in model_data.items() if hasattr(ModelConfig, k)})
        else:
            model_cfg = ModelConfig()

        valid_keys = {k: v for k, v in data.items() if hasattr(cls, k)}
        return cls(model=model_cfg, **valid_keys)

    def to_json(self, path: str, indent: int = 2) -> None:
        """Save configuration to a JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        data = asdict(self)
        with open(path, "w") as f:
            json.dump(data, f, indent=indent)
