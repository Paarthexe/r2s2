# R2S2: Recursive Residual Speech Separation

R2S2 is a deep learning system for multi-speaker speech separation.
The system uses a Dual-Path Recurrent Neural Network (DPRNN) and One-and-Rest Permutation Invariant Training (OR-PIT).
It separates one speaker at a time from an audio mixture.
This recursive process lets the model separate an arbitrary number of speakers with a fixed output size.

## Results

- **Validation SI-SDRi**: 4.13 dB

## Method Description

The neural network takes a raw mixture waveform $x(t)$ as input.
It calculates two output signals:
1. **Target speaker**: The separated speech of one source.
2. **Residual mixture**: The sum of all remaining sources.

During inference, the model operates recursively.
The residual output from step $k$ becomes the mixture input for step $k+1$.
Separation stops when the residual RMS energy is less than 0.01 or after 6 iterations.

### Architecture

- **Encoder**: 1D convolution with 64 filters, kernel size 16, stride 8, and ReLU activation.
- **Separator**: 6 DPRNN blocks that process feature chunks of size 100 with a 50 percent hop size. Each block contains an intra-chunk bidirectional LSTM, an inter-chunk bidirectional LSTM, hidden dimension 128, and LayerNorm.
- **Mask Estimation**: Sigmoid masks applied to encoder outputs with vectorized overlap-add reconstruction.
- **Decoder**: 1D transposed convolution that projects features back to audio samples.

### Training Strategy

- **Curriculum Learning**: Epochs 1 to 10 train on 2-speaker mixtures. Epochs 11 to 20 gradually add 3-speaker and 4-speaker mixtures. Epochs 21 to 40 train on the full mixture distribution.
- **Recursive Unrolling**: Starting at epoch 20, 50 percent of training batches execute a second separation step on the detached residual signal.
- **Loss Functions**:
  - Scale-Invariant SDR (SI-SDR) with One-and-Rest PIT.
  - Multi-resolution STFT loss at FFT sizes 512, 1024, and 2048 with weight warmup from 0.01 to 0.1 between epochs 8 and 15.
  - Mixture consistency L1 loss with weight 0.1.

## Training Parameters

| Parameter | Value |
|---|---|
| Sample rate | 16,000 Hz |
| Segment length | 4.0 seconds (64,000 samples) |
| Batch size | 16 (train), 64 (validation) |
| Encoder filters / kernel / stride | 64 / 16 / 8 |
| DPRNN blocks / hidden size | 6 / 128 |
| Chunk size / hop | 100 / 50 |
| Optimizer | AdamW (lr=1e-3, weight decay=1e-2) |
| Warmup steps | 400 |
| Learning rate scheduler | ReduceLROnPlateau (factor 0.5, patience 3) |
| Mixed precision | PyTorch AMP (autocast and GradScaler) |
| Gradient clipping | 5.0 |
| STFT loss weight | 0.01 to 0.1 (warmup epochs 8 to 15) |
| Consistency loss weight | 0.1 |
| Unroll start epoch / probability | Epoch 20 / 0.5 |
| RMS stop threshold | 0.01 |
| Maximum recursive iterations | 6 |

## Repository Structure

- `models/`: Model architecture (Encoder, Decoder, DPRNN separator, OR-PIT model) and loss functions.
- `training/`: Dataloaders with curriculum sampling, training engine with recursive unrolling, and checkpoint manager.
- `inference/`: Recursive separation pipeline and audio file export functions.
- `data_prep/`: Synthetic multi-speaker dataset generator using LibriSpeech.
- `configs/`: JSON configuration files for model, training, and inference.
- `train.py`: Command-line interface to train the model.
- `evaluate.py`: Command-line interface to run recursive speech separation.
- `generate.py`: Command-line interface to generate synthetic mixtures.
- `tests/`: Automated unit tests.

## Setup and Usage

### Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Dataset Generation

Generate the conversational speech dataset:

```bash
# Generate the dataset with default settings
python generate.py

# Specify an output directory and runtime limit
python generate.py --output-dir ./conversational_dataset --max-runtime-hours 5.0 --num-workers 8

# Generate a small sample set for testing
python generate.py --splits train val --max-samples 100 --output-dir ./test_dataset
```

### Dataset Directory Layout

The dataset loader requires audio organized in this directory structure:

```text
dataset_dir/
├── train/
│   ├── sample_0000000/
│   │   ├── source_1.flac
│   │   ├── source_2.flac
│   │   └── metadata.json
│   └── sample_0000001/
└── val/
    └── sample_0000000/
        ├── source_1.flac
        └── source_2.flac
```

Audio files must be single-channel mono sampled at 16 kHz.

### Model Training

Start model training with the configuration file:

```bash
python train.py --config configs/train_config.json
```

You can override parameters directly from the command line:

```bash
python train.py --epochs 20 --batch-size 8 --device cuda
```

### Recursive Separation

Separate speakers from an audio mixture file:

```bash
python evaluate.py --checkpoint checkpoints/best.pt --input-audio mixture.flac --output-dir ./separated
```

You can also run separation directly in Python:

```python
import soundfile as sf
from inference.pipeline import load_model_for_inference, recursive_separate

# Load the trained model checkpoint
model = load_model_for_inference("checkpoints/best.pt")

# Read the mixture audio file (16 kHz mono)
mixture, sample_rate = sf.read("mixture.flac", dtype="float32")

# Recursively separate the speakers
speakers = recursive_separate(model, mixture, rms_threshold=0.01, max_iters=6)

# Save each separated speaker to disk
for index, speaker_audio in enumerate(speakers):
    sf.write(f"speaker_{index + 1}.wav", speaker_audio, sample_rate)
```