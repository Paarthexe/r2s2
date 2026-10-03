import os
import json
from typing import Dict, Any


def load_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """Loads checkpoint state from JSON file or returns fresh default state."""
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load checkpoint from {checkpoint_path} ({e}). Creating new.")

    return {
        "train": {"count": 0, "elapsed_hours": 0.0},
        "val": {"count": 0, "elapsed_hours": 0.0},
        "test": {"count": 0, "elapsed_hours": 0.0},
        "global_seed_offset": 0,
    }


def save_checkpoint(state: Dict[str, Any], checkpoint_path: str) -> None:
    """Safely saves generation progress state to a JSON file."""
    os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
    temp_path = f"{checkpoint_path}.tmp"
    with open(temp_path, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(temp_path, checkpoint_path)
