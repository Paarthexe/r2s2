import os
import glob
import math
import random
from typing import Tuple, List, Optional
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader, Sampler

from training.config import TrainConfig


class SeparationDataset(Dataset):
    """Audio dataset for multi-speaker speech separation.

    Loads synthesized mixture directories containing individual source FLAC files.
    """

    def __init__(
        self,
        root: str,
        split: str,
        crop_samples: int = 64000,
        is_train: Optional[bool] = None,
    ):
        self.root = root
        self.split = split
        self.crop = crop_samples
        self.is_train = (split == "train") if is_train is None else is_train

        sample_dirs = sorted(glob.glob(os.path.join(root, split, "sample_*")))

        self.index: List[str] = []
        self.speaker_counts: List[int] = []
        for d in sample_dirs:
            sp = glob.glob(os.path.join(d, "source_*.flac"))
            if len(sp) >= 2:
                self.index.append(d)
                self.speaker_counts.append(len(sp))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        sample_dir = self.index[idx]
        source_paths = sorted(glob.glob(os.path.join(sample_dir, "source_*.flac")))
        n = len(source_paths)

        sources = []
        for sp in source_paths:
            audio, _ = sf.read(sp, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            sources.append(audio)

        length = min(len(s) for s in sources)
        sources = [s[:length] for s in sources]

        if length < self.crop:
            repeats = math.ceil(self.crop / length)
            sources = [np.tile(s, repeats)[: self.crop] for s in sources]
        elif self.is_train:
            start = random.randint(0, length - self.crop)
            sources = [s[start : start + self.crop] for s in sources]
        else:
            start = (length - self.crop) // 2
            sources = [s[start : start + self.crop] for s in sources]

        sources_arr = np.stack(sources, axis=0)
        target_idx = random.randint(0, n - 1) if self.is_train else idx % n
        target = sources_arr[target_idx]
        residual = sources_arr[np.arange(n) != target_idx].sum(axis=0)
        mixture = sources_arr.sum(axis=0)

        if self.is_train:
            max_speakers = 4
            padded_sources = np.zeros((max_speakers, self.crop), dtype=np.float32)
            padded_sources[:n] = sources_arr
            return (
                torch.from_numpy(mixture).float(),
                torch.from_numpy(target).float(),
                torch.from_numpy(residual).float(),
                torch.from_numpy(padded_sources).float(),
                target_idx,
                n,
            )
        else:
            return (
                torch.from_numpy(mixture).float(),
                torch.from_numpy(target).float(),
                torch.from_numpy(residual).float(),
            )


class CurriculumSampler(Sampler):
    """Samples batches according to curriculum speaker count probabilities."""

    def __init__(self, dataset: SeparationDataset):
        super().__init__()
        self.dataset = dataset
        self.num_samples = len(dataset)
        self.weights = torch.ones(self.num_samples)

    def set_weights(self, weights: torch.Tensor) -> None:
        self.weights = weights

    def __iter__(self):
        if self.num_samples == 0:
            return iter([])
        return iter(
            torch.multinomial(self.weights, self.num_samples, replacement=True).tolist()
        )

    def __len__(self) -> int:
        return self.num_samples


def get_curriculum_weights(
    epoch: int, total_epochs: int, speaker_counts: List[int]
) -> torch.Tensor:
    """Computes sample sampling weights based on curriculum schedule:

    - Epochs 1-10: 2 speakers only (weight 1.0 for 2-spk, 0.0 for 3/4-spk)
    - Epochs 11-20: Linear ramp of 3/4-speaker mixtures from 0.0 to 1.0
    - Epochs 20+: Full mixture distribution (weight 1.0 for all)
    """
    if epoch <= 10:
        w34 = 0.0
    elif epoch <= 20:
        w34 = (epoch - 10) / 10.0
    else:
        w34 = 1.0

    weights = [1.0 if n == 2 else w34 for n in speaker_counts]
    return torch.tensor(weights, dtype=torch.float)


def get_loaders(
    config: TrainConfig,
) -> Tuple[DataLoader, DataLoader, SeparationDataset]:
    """Builds training and validation DataLoaders with curriculum sampling."""
    train_ds = SeparationDataset(
        root=config.dataset_dir,
        split="train",
        crop_samples=config.crop_samples,
        is_train=True,
    )
    val_ds = SeparationDataset(
        root=config.dataset_dir,
        split="val",
        crop_samples=config.crop_samples,
        is_train=False,
    )

    base_kwargs = {
        "num_workers": config.num_workers,
        "pin_memory": (config.resolved_device == "cuda"),
    }
    if config.num_workers > 0:
        base_kwargs.update({"persistent_workers": True, "prefetch_factor": 2})

    train_sampler = CurriculumSampler(train_ds)
    train_dl = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        sampler=train_sampler,
        drop_last=True,
        **base_kwargs,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=config.batch_size * 4,
        shuffle=False,
        **base_kwargs,
    )

    return train_dl, val_dl, train_ds
