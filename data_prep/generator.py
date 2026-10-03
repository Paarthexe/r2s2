import os
import time
from typing import Dict, Any, Optional, List
from tqdm.auto import tqdm
from joblib import Parallel, delayed

from data_prep.config import DatasetConfig
from data_prep.loader import load_librispeech_split
from data_prep.checkpoint import load_checkpoint, save_checkpoint
from data_prep.pipeline import process_sample


def _worker_wrapper(args):
    """Worker function for joblib multiprocessing."""
    split_name, ds, spk_to_idx, available_speakers, sample_idx, seed, config = args
    return process_sample(
        split_name=split_name,
        ds=ds,
        spk_to_idx=spk_to_idx,
        available_speakers=available_speakers,
        sample_idx=sample_idx,
        seed=seed,
        config=config,
    )


class DatasetGenerator:
    """Manages stateful, parallel, chunked synthesis of multi-speaker speech datasets."""

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.checkpoint_path = config.execution.checkpoint_path
        self.state = load_checkpoint(self.checkpoint_path)

    def generate_split(
        self,
        split_name: str,
        allocated_hours: Optional[float] = None,
        max_samples: Optional[int] = None,
    ) -> None:
        """Generates samples for a specific dataset split until time or count budget is met."""
        if allocated_hours is None:
            allocated_hours = (
                self.config.execution.time_split_ratios.get(split_name, 0.1)
                * self.config.execution.max_runtime_hours
            )

        current_elapsed = self.state[split_name]["elapsed_hours"]
        current_count = self.state[split_name]["count"]

        if current_elapsed >= allocated_hours:
            print(f"\n{split_name} already completed its allocated time ({allocated_hours:.2f}h).")
            return

        if max_samples is not None and current_count >= max_samples:
            print(f"\n{split_name} already reached sample target ({current_count}/{max_samples}).")
            return

        print(
            f"\n--- Starting/Resuming {split_name} "
            f"(Allocated: {allocated_hours:.2f}h, Elapsed: {current_elapsed:.2f}h, Count: {current_count}) ---"
        )

        split_dir = os.path.join(self.config.execution.output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        # Lazy load dataset split
        ds, spk_to_idx, available_speakers = load_librispeech_split(split_name)

        chunk_size = self.config.execution.chunk_size
        num_workers = self.config.execution.num_workers
        start_time = time.time()

        while self.state[split_name]["elapsed_hours"] < allocated_hours:
            if max_samples is not None and self.state[split_name]["count"] >= max_samples:
                break

            current_count = self.state[split_name]["count"]
            seed_offset = self.state["global_seed_offset"]
            this_chunk_size = chunk_size
            if max_samples is not None:
                this_chunk_size = min(chunk_size, max_samples - current_count)

            args_list = []
            for i in range(this_chunk_size):
                idx = current_count + i
                seed = self.config.execution.seed + seed_offset + i
                args_list.append(
                    (
                        split_name,
                        ds,
                        spk_to_idx,
                        available_speakers,
                        idx,
                        seed,
                        self.config,
                    )
                )

            results = Parallel(n_jobs=num_workers, backend="loky")(
                delayed(_worker_wrapper)(arg)
                for arg in tqdm(args_list, desc=f"{split_name} (Chunk)")
            )

            successful = sum(1 for res in results if res is not None)
            elapsed_chunk_hours = (time.time() - start_time) / 3600.0

            self.state[split_name]["count"] += successful
            self.state[split_name]["elapsed_hours"] += elapsed_chunk_hours
            self.state["global_seed_offset"] += this_chunk_size
            save_checkpoint(self.state, self.checkpoint_path)

            start_time = time.time()
            time_left = max(0.0, allocated_hours - self.state[split_name]["elapsed_hours"])
            print(
                f"  -> Generated {successful}/{this_chunk_size} in chunk. "
                f"Split total: {self.state[split_name]['count']}. Time left: {time_left:.2f}h"
            )

        print(f"Finished {split_name} generation.")

    def run(
        self,
        splits: Optional[List[str]] = None,
        max_runtime_hours: Optional[float] = None,
        max_samples_per_split: Optional[int] = None,
    ) -> None:
        """Executes generation across all requested splits."""
        if splits is None:
            splits = ["train", "val", "test"]

        if max_runtime_hours is not None:
            self.config.execution.max_runtime_hours = max_runtime_hours

        for split in splits:
            allocated = (
                self.config.execution.time_split_ratios.get(split, 0.1)
                * self.config.execution.max_runtime_hours
            )
            self.generate_split(
                split_name=split,
                allocated_hours=allocated,
                max_samples=max_samples_per_split,
            )

        print("\nAll requested splits completed!")
