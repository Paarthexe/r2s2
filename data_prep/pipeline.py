import os
import io
import json
from typing import Optional, Dict, Any, Tuple, List
import numpy as np
import soundfile as sf
import librosa

from data_prep.config import DatasetConfig
from data_prep.audio import schedule_speakers, calculate_rms_db


def process_sample(
    split_name: str,
    ds: Any,
    spk_to_idx: Dict[int, List[int]],
    available_speakers: List[int],
    sample_idx: int,
    seed: int,
    config: DatasetConfig,
) -> Optional[Dict[str, Any]]:
    """Synthesizes a single multi-speaker conversational mixture with scheduled overlap.

    Returns the sample metadata dict if successful, or None if validation/scheduling fails.
    """
    try:
        srng = np.random.default_rng(seed)
        audio_cfg = config.audio
        spk_cfg = config.speakers
        sched_cfg = config.scheduling
        target_sr = audio_cfg.sample_rate

        for _ in range(10):
            num_speakers = int(srng.choice(spk_cfg.counts, p=spk_cfg.probabilities))
            speakers = [
                int(s)
                for s in srng.choice(available_speakers, size=num_speakers, replace=False)
            ]

            signals: List[np.ndarray] = []
            utt_ids: List[str] = []
            durations: List[float] = []
            gains: List[float] = []

            valid_selection = True
            for spk in speakers:
                indices = spk_to_idx[spk]
                for _ in range(10):
                    utt_idx = int(srng.choice(indices))
                    item = ds[utt_idx]
                    audio, orig_sr = sf.read(io.BytesIO(item["audio"]["bytes"]))

                    # Minimum length check
                    min_samples_orig = audio_cfg.min_utterance_samples * (orig_sr / target_sr)
                    if len(audio) < min_samples_orig:
                        continue

                    # Resample if needed
                    if orig_sr != target_sr:
                        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)

                    # Minimum RMS check
                    rms_db = calculate_rms_db(audio)
                    if rms_db < audio_cfg.min_rms_db:
                        continue

                    # Gain augmentation
                    gain_db = float(srng.uniform(*audio_cfg.gain_range_db))
                    audio = audio * (10.0 ** (gain_db / 20.0))

                    signals.append(audio)
                    durations.append(len(audio) / target_sr)
                    gains.append(gain_db)

                    # Extract utterance id
                    raw_id = item.get("id", item.get("file", f"{spk}_{utt_idx}"))
                    clean_id = (
                        str(raw_id)
                        .split("/")[-1]
                        .replace(".flac", "")
                        .replace(".wav", "")
                    )
                    utt_ids.append(clean_id)
                    break
                else:
                    valid_selection = False
                    break

            if not valid_selection:
                continue

            start_times, overlap = schedule_speakers(
                durations,
                srng,
                min_overlap=sched_cfg.min_overlap,
                max_overlap=sched_cfg.max_overlap,
                min_start_delay=sched_cfg.min_start_delay,
                max_schedule_attempts=sched_cfg.max_schedule_attempts,
                max_placement_attempts=sched_cfg.max_placement_attempts,
            )
            if start_times is None or overlap > sched_cfg.max_allowed_overlap:
                continue

            mix_duration = max(s + d for s, d in zip(start_times, durations))
            mix_samples = int(np.ceil(mix_duration * target_sr))

            sources = [np.zeros(mix_samples, dtype=np.float32) for _ in range(num_speakers)]
            for i, (audio, start) in enumerate(zip(signals, start_times)):
                start_samp = int(start * target_sr)
                end_samp = start_samp + len(audio)
                sources[i][start_samp:end_samp] = audio

            mixture = np.sum(sources, axis=0)

            peak = float(np.max(np.abs(mixture)))
            if peak > 0.95:
                scale = 0.95 / peak
                mixture *= scale
                for i in range(num_speakers):
                    sources[i] *= scale
            else:
                scale = 1.0

            mix_name = f"sample_{sample_idx:07d}"
            sample_dir = os.path.join(config.execution.output_dir, split_name, mix_name)
            os.makedirs(sample_dir, exist_ok=True)

            sf.write(
                os.path.join(sample_dir, "mixture.flac"),
                mixture,
                target_sr,
                format="FLAC",
            )
            for i in range(num_speakers):
                sf.write(
                    os.path.join(sample_dir, f"source_{i+1}.flac"),
                    sources[i],
                    target_sr,
                    format="FLAC",
                )

            metadata = {
                "mixture_id": mix_name,
                "speaker_ids": speakers,
                "utterance_ids": utt_ids,
                "num_speakers": num_speakers,
                "start_times": [round(float(s), 4) for s in start_times],
                "durations": [round(float(d), 4) for d in durations],
                "gains_db": [round(float(g), 4) for g in gains],
                "overlap_ratio": round(float(overlap), 4),
                "scale_factor": round(float(scale), 6),
                "mix_duration": round(float(mix_duration), 4),
            }

            with open(os.path.join(sample_dir, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)

            return metadata

        return None

    except Exception:
        return None
