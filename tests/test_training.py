import os
import soundfile as sf
import numpy as np
import torch
import pytest
from torch.utils.data import DataLoader

from training.config import TrainConfig
from training.dataset import SeparationDataset, CurriculumSampler, get_curriculum_weights
from training.engine import run_epoch
from models.config import ModelConfig
from models.separator import ORPIT_Model


def create_synthetic_dataset(base_dir):
    """Creates a small mock dataset directory for testing."""
    for split in ["train", "val"]:
        split_dir = os.path.join(base_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        for i in range(2):
            sample_dir = os.path.join(split_dir, f"sample_{i:05d}")
            os.makedirs(sample_dir, exist_ok=True)
            # Create 2 source flacs
            for s in range(1, 3):
                wav = np.random.randn(8000).astype(np.float32) * 0.1
                sf.write(os.path.join(sample_dir, f"source_{s}.flac"), wav, 16000)


def test_separation_dataset(tmp_path):
    create_synthetic_dataset(str(tmp_path))
    ds = SeparationDataset(root=str(tmp_path), split="train", crop_samples=4000)
    assert len(ds) == 2

    sample = ds[0]
    # In train: (mixture, target, residual, padded_sources, target_idx, n)
    assert len(sample) == 6
    mix, target, residual, padded, target_idx, n = sample
    assert mix.shape == (4000,)
    assert target.shape == (4000,)
    assert residual.shape == (4000,)
    assert padded.shape == (4, 4000)
    assert n == 2


def test_curriculum_weights():
    speaker_counts = [2, 3, 4]
    w_early = get_curriculum_weights(epoch=5, total_epochs=40, speaker_counts=speaker_counts)
    assert w_early[0] == 1.0
    assert w_early[1] == 0.0
    assert w_early[2] == 0.0

    w_mid = get_curriculum_weights(epoch=15, total_epochs=40, speaker_counts=speaker_counts)
    assert w_mid[0] == 1.0
    assert w_mid[1] == 0.5
    assert w_mid[2] == 0.5

    w_late = get_curriculum_weights(epoch=25, total_epochs=40, speaker_counts=speaker_counts)
    assert torch.all(w_late == 1.0)


def test_training_step(tmp_path):
    create_synthetic_dataset(str(tmp_path))
    cfg = TrainConfig(
        dataset_dir=str(tmp_path),
        crop_seconds=0.25,  # 4000 samples
        batch_size=2,
        epochs=1,
        num_workers=0,
        device="cpu",
        model=ModelConfig(enc_filters=16, dprnn_hidden=16, dprnn_layers=1, chunk_size=20),
    )

    ds = SeparationDataset(root=str(tmp_path), split="train", crop_samples=cfg.crop_samples)
    dl = DataLoader(ds, batch_size=cfg.batch_size)

    model = ORPIT_Model(config=cfg.model).to("cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cpu", enabled=False)

    loss, metrics, step, unrolls = run_epoch(
        model=model,
        loader=dl,
        optimizer=optimizer,
        scaler=scaler,
        is_train=True,
        stft_weight=0.01,
        config=cfg,
        epoch=1,
        global_step=0,
    )

    assert isinstance(loss, float)
    assert "sisdr" in metrics
    assert "stft" in metrics
    assert "consistency" in metrics
    assert step == 1
