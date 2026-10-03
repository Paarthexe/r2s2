import os
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import yaml
import multiprocessing


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    min_utterance_sec: float = 3.0
    min_rms_db: float = -25.0
    gain_range_db: Tuple[float, float] = (-4.0, 4.0)

    @property
    def min_utterance_samples(self) -> int:
        return int(self.sample_rate * self.min_utterance_sec)


@dataclass
class SpeakerConfig:
    counts: List[int] = field(default_factory=lambda: [2, 3, 4])
    probabilities: List[float] = field(default_factory=lambda: [0.70, 0.20, 0.10])


@dataclass
class SchedulingConfig:
    min_overlap: float = 0.30
    max_overlap: float = 0.70
    max_allowed_overlap: float = 0.75
    min_start_delay: float = 0.5
    max_schedule_attempts: int = 50
    max_placement_attempts: int = 20


@dataclass
class ExecutionConfig:
    seed: int = 42
    num_workers: int = field(default_factory=lambda: max(1, multiprocessing.cpu_count() - 1))
    chunk_size: int = 500
    max_runtime_hours: float = 10.0
    output_dir: str = "./conversational_dataset_v2"
    checkpoint_filename: str = "checkpoint.json"
    time_split_ratios: Dict[str, float] = field(
        default_factory=lambda: {"train": 0.8, "val": 0.1, "test": 0.1}
    )

    @property
    def checkpoint_path(self) -> str:
        return os.path.join(self.output_dir, self.checkpoint_filename)


@dataclass
class DatasetConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    speakers: SpeakerConfig = field(default_factory=SpeakerConfig)
    scheduling: SchedulingConfig = field(default_factory=SchedulingConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "DatasetConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}

        audio_data = data.get("audio", {})
        speaker_data = data.get("speakers", {})
        sched_data = data.get("scheduling", {})
        exec_data = data.get("execution", {})

        if "gain_range_db" in audio_data:
            audio_data["gain_range_db"] = tuple(audio_data["gain_range_db"])

        if exec_data.get("num_workers") is None:
            exec_data["num_workers"] = max(1, multiprocessing.cpu_count() - 1)

        return cls(
            audio=AudioConfig(**audio_data),
            speakers=SpeakerConfig(**speaker_data),
            scheduling=SchedulingConfig(**sched_data),
            execution=ExecutionConfig(**exec_data),
        )
