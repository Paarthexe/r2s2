import os
import json
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any


@dataclass
class ModelConfig:
    """Hyperparameters for the R2S2 Dual-Path RNN separation network."""

    enc_filters: int = 64
    enc_kernel: int = 16
    enc_stride: int = 8
    dprnn_hidden: int = 128
    dprnn_layers: int = 6
    chunk_size: int = 100

    @classmethod
    def from_json(cls, path: str) -> "ModelConfig":
        """Load configuration from a JSON file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file not found: {path}")
        with open(path, "r") as f:
            data = json.load(f)
        # Support optional nested 'model' key or top-level keys
        model_data = data.get("model", data)
        valid_keys = {k: v for k, v in model_data.items() if hasattr(cls, k)}
        return cls(**valid_keys)

    def to_json(self, path: str, indent: int = 2) -> None:
        """Save configuration to a JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=indent)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
