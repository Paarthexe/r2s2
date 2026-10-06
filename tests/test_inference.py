import os
import soundfile as sf
import numpy as np
import torch
import pytest

from models.config import ModelConfig
from models.separator import ORPIT_Model
from inference.pipeline import rms, recursive_separate, separate_audio_file


def test_rms():
    # Pure sine wave with amp 1.0 has RMS ~ 1/sqrt(2) ~ 0.707
    t = torch.linspace(0, 1, 16000)
    sig = torch.sin(2 * np.pi * 440 * t)
    val = rms(sig)
    assert np.isclose(val, 0.707, atol=0.01)


def test_recursive_separate():
    cfg = ModelConfig(enc_filters=16, dprnn_hidden=16, dprnn_layers=1, chunk_size=20)
    model = ORPIT_Model(config=cfg).eval()

    mix = np.random.randn(8000).astype(np.float32) * 0.1
    # Run peeling with max 3 iters
    speakers = recursive_separate(model=model, mixture_np=mix, rms_threshold=0.001, max_iters=3, device="cpu")

    assert len(speakers) >= 1
    assert len(speakers) <= 3
    assert len(speakers[0]) == 8000


def test_separate_audio_file(tmp_path):
    cfg = ModelConfig(enc_filters=16, dprnn_hidden=16, dprnn_layers=1, chunk_size=20)
    model = ORPIT_Model(config=cfg).eval()

    input_file = str(tmp_path / "mix.wav")
    out_dir = str(tmp_path / "separated")

    audio = np.random.randn(8000).astype(np.float32) * 0.1
    sf.write(input_file, audio, 16000)

    out_files = separate_audio_file(
        model=model,
        input_path=input_file,
        output_dir=out_dir,
        target_sr=16000,
        rms_threshold=0.001,
        max_iters=2,
        device="cpu",
    )

    assert len(out_files) >= 1
    for f in out_files:
        assert os.path.exists(f)
