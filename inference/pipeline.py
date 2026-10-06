import os
from typing import List, Optional, Tuple
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import librosa

from models.config import ModelConfig
from models.separator import ORPIT_Model


def load_model_for_inference(
    checkpoint_path: str,
    config: Optional[ModelConfig] = None,
    device: Optional[str] = None,
) -> nn.Module:
    """Loads trained R2S2 model weights from checkpoint into evaluation mode."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if config is None:
        config = ModelConfig()

    model = ORPIT_Model(config=config).to(device)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["model"] if "model" in ckpt else ckpt
    clean_state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(clean_state)
    model.eval()
    return model


def rms(signal: torch.Tensor) -> float:
    """Calculates root-mean-square energy of audio tensor."""
    return signal.pow(2).mean().sqrt().item()


def recursive_separate(
    model: nn.Module,
    mixture_np: np.ndarray,
    rms_threshold: float = 0.01,
    max_iters: int = 6,
    device: Optional[str] = None,
) -> List[np.ndarray]:
    """Recursively separates speakers from an acoustic mixture using one-and-rest peeling.

    Args:
        model: Trained R2S2 / ORPIT_Model.
        mixture_np: 1D audio waveform array of shape (T,).
        rms_threshold: Stop peeling when residual RMS drops below this threshold.
        max_iters: Maximum number of speaker extraction iterations.
        device: Torch device to execute on.

    Returns:
        List of 1D numpy arrays, one for each separated speaker.
    """
    if device is None:
        device = next(model.parameters()).device

    mix = torch.from_numpy(mixture_np).float().unsqueeze(0).to(device)
    speakers: List[np.ndarray] = []

    with torch.no_grad():
        for _ in range(max_iters):
            out = model(mix)
            target = out[0, 0]
            residual = out[0, 1]

            speakers.append(target.cpu().numpy())

            if rms(residual) < rms_threshold:
                break

            mix = residual.unsqueeze(0)

    return speakers


def separate_audio_file(
    model: nn.Module,
    input_path: str,
    output_dir: str,
    target_sr: int = 16000,
    rms_threshold: float = 0.01,
    max_iters: int = 6,
    device: Optional[str] = None,
) -> List[str]:
    """Separates an audio file and saves each extracted source waveform to disk."""
    os.makedirs(output_dir, exist_ok=True)

    audio, sr = sf.read(input_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    speakers = recursive_separate(
        model=model,
        mixture_np=audio,
        rms_threshold=rms_threshold,
        max_iters=max_iters,
        device=device,
    )

    output_files = []
    for idx, spk in enumerate(speakers):
        out_file = os.path.join(output_dir, f"speaker_{idx + 1}.wav")
        sf.write(out_file, spk, target_sr)
        output_files.append(out_file)

    return output_files
