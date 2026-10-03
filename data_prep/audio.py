from typing import List, Tuple, Optional
import numpy as np


def compute_overlap_ratio(start_times: List[float], end_times: List[float]) -> float:
    """Computes the ratio of time where 2 or more speakers are talking concurrently

    over the total duration during which at least one speaker is active.
    """
    events = []
    for s, e in zip(start_times, end_times):
        events.append((s, 1))
        events.append((e, -1))
    events.sort()

    active_speakers = 0
    total_speech_time = 0.0
    overlap_time = 0.0
    last_t = events[0][0]

    for t, delta in events:
        dur = t - last_t
        if active_speakers > 0:
            total_speech_time += dur
        if active_speakers > 1:
            overlap_time += dur

        active_speakers += delta
        last_t = t

    return overlap_time / total_speech_time if total_speech_time > 0 else 0.0


def schedule_speakers(
    durations: List[float],
    rng: np.random.Generator,
    min_overlap: float = 0.30,
    max_overlap: float = 0.70,
    min_start_delay: float = 0.5,
    max_schedule_attempts: int = 50,
    max_placement_attempts: int = 20,
) -> Tuple[Optional[List[float]], float]:
    """Attempts to place speakers sequentially such that the overall overlap ratio

    satisfies the min/max overlap criteria and start times have minimal delay.
    """
    num_speakers = len(durations)

    for _ in range(max_schedule_attempts):
        start_times = [0.0]
        end_times = [durations[0]]
        valid = True

        for i in range(1, num_speakers):
            max_start = max(end_times)
            placed = False
            for _ in range(max_placement_attempts):
                s = rng.uniform(0, max_start)
                if all(abs(s - st) >= min_start_delay for st in start_times):
                    start_times.append(s)
                    end_times.append(s + durations[i])
                    placed = True
                    break
            if not placed:
                valid = False
                break

        if not valid:
            continue

        overlap = compute_overlap_ratio(start_times, end_times)
        if min_overlap <= overlap <= max_overlap:
            return start_times, overlap

    return None, 0.0


def calculate_rms_db(audio: np.ndarray) -> float:
    """Calculates RMS energy in decibels for an audio array."""
    rms = np.sqrt(np.mean(audio**2))
    return 20.0 * np.log10(max(rms, 1e-10))
