from collections import defaultdict
from typing import Tuple, List, Dict, Any
from datasets import load_dataset, Audio


SPLIT_NAME_MAP = {
    "train": "train.100",
    "train.100": "train.100",
    "train.360": "train.360",
    "val": "validation",
    "validation": "validation",
    "test": "test",
}


def load_librispeech_split(
    split_name: str,
    dataset_name: str = "openslr/librispeech_asr",
    config_name: str = "clean",
) -> Tuple[Any, Dict[int, List[int]], List[int]]:
    """Loads a LibriSpeech split with undecoded audio bytes and creates a speaker-to-indices mapping."""
    canonical_split = SPLIT_NAME_MAP.get(split_name, split_name)
    ds = load_dataset(dataset_name, config_name, split=canonical_split)
    ds = ds.cast_column("audio", Audio(decode=False))

    speaker_to_indices = defaultdict(list)
    for idx, spk_id in enumerate(ds["speaker_id"]):
        speaker_to_indices[spk_id].append(idx)

    speakers_list = list(speaker_to_indices.keys())
    print(f"Loaded {split_name} ({canonical_split}): {len(ds)} utterances from {len(speakers_list)} speakers")
    return ds, speaker_to_indices, speakers_list
