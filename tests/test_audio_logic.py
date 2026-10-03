import numpy as np
import pytest

from data_prep.config import DatasetConfig
from data_prep.audio import (
    compute_overlap_ratio,
    schedule_speakers,
    calculate_rms_db,
)


def test_compute_overlap_ratio():
    # Two identical concurrent intervals: [0, 4] and [0, 4] -> 100% overlap
    start_times = [0.0, 0.0]
    end_times = [4.0, 4.0]
    overlap = compute_overlap_ratio(start_times, end_times)
    assert np.isclose(overlap, 1.0)

    # Disjoint intervals: [0, 2] and [3, 5] -> 0% overlap
    start_times = [0.0, 3.0]
    end_times = [2.0, 5.0]
    overlap = compute_overlap_ratio(start_times, end_times)
    assert np.isclose(overlap, 0.0)

    # Partial overlap: [0, 3] and [1, 4]
    # Total speech: 4.0, overlap [1, 3]: 2.0 -> 2/4 = 0.5
    start_times = [0.0, 1.0]
    end_times = [3.0, 4.0]
    overlap = compute_overlap_ratio(start_times, end_times)
    assert np.isclose(overlap, 0.5)


def test_schedule_speakers():
    rng = np.random.default_rng(42)
    durations = [4.0, 4.0]
    start_times, overlap = schedule_speakers(
        durations, rng, min_overlap=0.2, max_overlap=0.8, min_start_delay=0.5
    )
    assert start_times is not None
    assert len(start_times) == 2
    assert 0.2 <= overlap <= 0.8
    assert abs(start_times[1] - start_times[0]) >= 0.5


def test_calculate_rms_db():
    sine = np.sin(np.linspace(0, 2 * np.pi * 100, 16000))
    rms_db = calculate_rms_db(sine)
    # Pure sine wave RMS is 1/sqrt(2) ~ 0.7071 -> 20*log10(0.7071) ~ -3.01 dB
    assert np.isclose(rms_db, -3.01, atol=0.1)


def test_config_loading():
    config = DatasetConfig.from_yaml("configs/dataset_config.yaml")
    assert config.audio.sample_rate == 16000
    assert config.speakers.counts == [2, 3, 4]
    assert config.scheduling.min_overlap == 0.30
    assert config.execution.seed == 42
